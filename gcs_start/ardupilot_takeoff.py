#!/usr/bin/env python3
"""ArduPilot take-off procedure shared by every controller in this package.

This is the start-up sequence of controller.py as flown on Friday 2026-11-06
(pole orbit, five laps, LAND), lifted out so that every node starts a flight the
same way:

    1. WAIT_ARM  nothing is asked of the FCU until the PILOT has armed with the
                 transmitter. This module has no arming client: it cannot arm.
                 "Armed" needs a fresh /mavros/state (< 3 s old) and a real local
                 pose; the pose z at that moment is the ground reference (local z
                 is EKF-origin relative, it is not 0 on the pad).
    2. GUIDED    request GUIDED, wait for the FCU to report it.
    3. SETTLE    wait until settle_s (5 s) after arming.
    4. TAKEOFF   /mavros/cmd/takeoff to the requested altitude; asked again if
                 the FCU refuses it.
    5. CLIMB     wait until z is within reach_margin (0.5 m) of the altitude
                 above the ground reference.
    6. AIRBORNE  on_airborne() is called once. Only now may the owner fly to a
                 position or start its control law.

Disarming at any point before AIRBORNE (pilot changed their mind, ArduPilot's
idle auto-disarm) returns to step 1 and the whole sequence, including the 5 s,
runs again after the next arming. Once the FCU has reported GUIDED, a change to
any other mode is the pilot's: the module never takes the mode back and sends
nothing while the FCU is out of GUIDED. A lost take-off reply does not stall the
sequence: being off the ground in GUIDED after a request counts as accepted.

Use:
    self.takeoff = PilotArmedTakeoff(self, altitude=2.0,
                                     on_armed=self._seed_estimator,
                                     on_airborne=self._start_control_law)
A node that starts on a button rather than at launch passes autostart=False and
calls start(altitude) / cancel().
"""
from rclpy.qos import (QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       DurabilityPolicy)
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandTOL, SetMode


class PilotArmedTakeoff:
    IDLE = 'idle'
    WAIT_ARM = 'wait_arm'
    GUIDED = 'guided'
    SETTLE = 'settle'
    TAKEOFF = 'takeoff'
    CLIMB = 'climb'
    AIRBORNE = 'airborne'

    STATE_FRESH_S = 3.0      # a /mavros/state older than this proves nothing
    GUIDED_RETRY_S = 3.0     # ask for GUIDED again if the FCU has not switched
    TAKEOFF_RETRY_S = 1.0    # pause before asking again after a refusal
    TAKEOFF_ANSWER_S = 10.0  # no answer from the take-off service: ask again
    LIFTOFF_M = 0.5          # this far above the ground reference = it has left the ground

    def __init__(self, node, altitude, on_airborne=None, on_armed=None,
                 on_disarmed=None, settle_s=5.0, reach_margin=0.5,
                 autostart=True, armed_epoch=0.0):
        self.node = node
        self.altitude = float(altitude)
        self.settle_s = float(settle_s)
        self.reach_margin = float(reach_margin)
        self._on_airborne = on_airborne
        self._on_armed = on_armed
        self._on_disarmed = on_disarmed
        # When the pilot armed (epoch seconds), if whoever started this node knows
        # it (fpv_switch watches the FC). A node started AFTER arming would
        # otherwise count settle_s from the moment it first sees "armed", and
        # ArduPilot disarms by itself DISARM_DELAY (10 s) after arming on the
        # ground: arm + press Start + node start-up + 5 s does not fit. With the
        # true arming time the take-off is asked for settle_s after ARMING.
        # Only used while the FC has stayed armed since this node started.
        self._armed_epoch = float(armed_epoch)
        self._seen_disarmed = False

        self.phase = self.IDLE
        self.status = 'idle'
        self.ground_z = 0.0          # local z when the pilot armed
        self.armed_time = None
        self.airborne_time = None

        self._state = None
        self._state_t = 0.0
        self._z = 0.0
        self._pose_received = False
        self._guided_req_t = None
        self._tko_req_t = None
        self._tko_retry_t = 0.0
        self._guided_seen = False    # the FCU has reported GUIDED since this arming
        self._tko_asked = False      # a take-off has been requested since this arming
        self._said = None

        state_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                               history=HistoryPolicy.KEEP_LAST, depth=10)
        pose_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                              durability=DurabilityPolicy.VOLATILE,
                              history=HistoryPolicy.KEEP_LAST, depth=10)
        node.create_subscription(State, '/mavros/state', self._on_state, state_qos)
        node.create_subscription(PoseStamped, '/mavros/local_position/pose',
                                 self._on_pose, pose_qos)
        # Deliberately no /mavros/cmd/arming client.
        self._mode_client = node.create_client(SetMode, '/mavros/set_mode')
        self._takeoff_client = node.create_client(CommandTOL, '/mavros/cmd/takeoff')
        node.create_timer(0.2, self._tick)

        if autostart:
            self.start()

    # ------------------------------------------------------------- public
    @property
    def airborne(self):
        return self.phase == self.AIRBORNE

    @property
    def active(self):
        return self.phase not in (self.IDLE, self.AIRBORNE)

    def start(self, altitude=None):
        """Begin (or begin again) waiting for the pilot to arm."""
        if altitude is not None:
            self.altitude = float(altitude)
        self._to_wait_arm()
        self._say(f'Start-up: waiting for the PILOT to arm -> GUIDED -> '
                  f'{self.settle_s:.0f} s -> takeoff to {self.altitude:.1f} m. '
                  'This node never arms.', once=False)

    def cancel(self):
        """Stop the sequence. Sends nothing to the FCU."""
        self.phase = self.IDLE
        self.status = 'idle'

    # ----------------------------------------------------------- internals
    def _now(self):
        return self.node.get_clock().now().nanoseconds / 1e9

    def _say(self, text, level='info', once=True):
        self.status = text
        if once and text == self._said:
            return
        self._said = text
        # One call site per severity: rclpy refuses a call site whose severity changes.
        log = self.node.get_logger()
        if level == 'error':
            log.error(text)
        elif level == 'warn':
            log.warn(text)
        else:
            log.info(text)

    def _to_wait_arm(self):
        self.phase = self.WAIT_ARM
        self.armed_time = None
        self.airborne_time = None
        self._guided_req_t = None
        self._tko_req_t = None
        self._tko_retry_t = 0.0
        self._guided_seen = False
        self._tko_asked = False

    def _on_state(self, msg):
        self._state = msg
        self._state_t = self._now()

    def _on_pose(self, msg):
        self._z = msg.pose.position.z
        self._pose_received = True

    def _tick(self):
        if self.phase in (self.IDLE, self.AIRBORNE):
            return
        st = self._state
        if st is None or not st.connected:
            self._say('Waiting for the FCU connection (MAVROS).')
            return
        now = self._now()
        if now - self._state_t > self.STATE_FRESH_S:
            self._say('FCU state is stale - holding the start-up sequence.', 'warn')
            return

        if self.phase == self.WAIT_ARM:
            if not st.armed:
                self._seen_disarmed = True
                self._say(f'NOT ARMED: waiting for the pilot to arm with the '
                          f'transmitter (FCU mode {st.mode}). This node never arms.',
                          'warn')
                return
            if not self._pose_received:
                self._say('Armed; waiting for the first local pose before confirming.')
                return
            self.armed_time = now
            already = ''
            if self._armed_epoch > 0.0 and not self._seen_disarmed:
                age = now - self._armed_epoch
                if 0.0 <= age <= 60.0:      # a sane, recent arming time only
                    self.armed_time = self._armed_epoch
                    already = f' {age:.1f} s ago'
            self._seen_disarmed = True      # the hint is for this arming only
            self.ground_z = self._z
            self.phase = self.GUIDED
            self._say(f'Armed by the pilot{already} (mode {st.mode}). '
                      f'Ground altitude reference z={self.ground_z:.2f} m')
            if self._on_armed is not None:
                self._on_armed()
            return

        if not st.armed:
            self._say('Disarmed before the take-off completed - back to waiting '
                      'for the pilot to arm.', 'warn')
            self._to_wait_arm()
            if self._on_disarmed is not None:
                self._on_disarmed()
            return

        # Once the FCU has been in GUIDED for this arming, a different mode means
        # the PILOT changed it: never take the mode back, in any phase (2026-11-12:
        # the retry loop below re-requested GUIDED four times against the pilot's
        # RTL / LOITER). Nothing more is sent until GUIDED returns or it disarms.
        if self._guided_seen and st.mode != 'GUIDED':
            self._say(f'FCU is in {st.mode}, not GUIDED: the pilot has control. '
                      'Sending nothing.', 'warn')
            return

        # The take-off reply can be lost on the companion link (2026-11-12: the
        # FCU climbed to 5 m while MAVROS reported "ack timeout", and every
        # further request was refused because it was already flying). The
        # altitude is the truth: off the ground in GUIDED after a request = the
        # take-off is under way, whatever the service answered.
        if (self._tko_asked and self.phase in (self.SETTLE, self.TAKEOFF)
                and self._z - self.ground_z > self.LIFTOFF_M):
            self.phase = self.CLIMB
            self._say(f'Off the ground ({self._z - self.ground_z:.2f} m) without a '
                      'take-off reply - take-off is under way (monitoring altitude).',
                      'warn')

        if self.phase == self.GUIDED:
            if st.mode == 'GUIDED':
                self._guided_seen = True
                self.phase = self.SETTLE
                self._say('GUIDED confirmed by FCU.')
            elif self._guided_req_t is None or now - self._guided_req_t > self.GUIDED_RETRY_S:
                self._request_guided(now)
            return

        if self.phase == self.SETTLE:
            if st.mode != 'GUIDED':
                self.phase = self.GUIDED
                self._guided_req_t = None
                self._say(f'FCU left GUIDED ({st.mode}) before take-off - asking again.', 'warn')
                return
            if now - self.armed_time < self.settle_s:
                self._say(f'Armed + GUIDED. Take-off {self.settle_s:.0f} s after arming...')
                return
            if now >= self._tko_retry_t:
                self._request_takeoff(now)
            return

        if self.phase == self.TAKEOFF:
            if now - self._tko_req_t > self.TAKEOFF_ANSWER_S:
                self._say('No answer to the take-off request - asking again.', 'warn')
                self.phase = self.SETTLE
            return

        if self.phase == self.CLIMB:
            if st.mode != 'GUIDED':
                self._say(f'FCU is in {st.mode}, not GUIDED: the pilot has control. '
                          'Not starting the controller.', 'warn')
                return
            agl = self._z - self.ground_z
            if agl > self.altitude - self.reach_margin:
                self.phase = self.AIRBORNE
                self.airborne_time = now
                self._say(f'Takeoff complete at z={self._z:.2f} m '
                          f'({agl:.2f} m above the ground reference).')
                if self._on_airborne is not None:
                    self._on_airborne()

    def _request_guided(self, now):
        if not self._mode_client.service_is_ready():
            self._say('SetMode service not ready yet.', 'warn')
            return
        self._guided_req_t = now
        req = SetMode.Request()
        req.custom_mode = 'GUIDED'
        self._mode_client.call_async(req).add_done_callback(self._guided_done)
        self._say('Requesting GUIDED...', once=False)

    def _guided_done(self, fut):
        try:
            ok = bool(getattr(fut.result(), 'mode_sent', False))
        except Exception as e:
            self.node.get_logger().error(f'SetMode exception: {e}')
            ok = False
        if ok:
            self.node.get_logger().info('GUIDED command accepted (awaiting FCU report).')
        else:
            self.node.get_logger().error('GUIDED command rejected by FCU.')

    def _request_takeoff(self, now):
        if not self._takeoff_client.service_is_ready():
            self._say('Takeoff service not ready yet.', 'warn')
            return
        self.phase = self.TAKEOFF
        self._tko_req_t = now
        self._tko_asked = True
        req = CommandTOL.Request()
        req.altitude = float(self.altitude)
        self._takeoff_client.call_async(req).add_done_callback(self._takeoff_done)
        self._say(f'Requesting takeoff to {self.altitude:.1f} m...', once=False)

    def _takeoff_done(self, fut):
        try:
            ok = bool(getattr(fut.result(), 'success', False))
        except Exception as e:
            self.node.get_logger().error(f'Takeoff exception: {e}')
            ok = False
        if self.phase != self.TAKEOFF:
            return                      # disarmed / cancelled while waiting
        if ok:
            self.phase = self.CLIMB
            self._say('Takeoff command accepted (monitoring altitude).', once=False)
        else:
            self.phase = self.SETTLE
            self._tko_retry_t = self._now() + self.TAKEOFF_RETRY_S
            self._say('Takeoff rejected by FCU - asking again.', 'error', once=False)
