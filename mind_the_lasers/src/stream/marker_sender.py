import socket


# Numeric codes sent over UDP for the Unicorn BCI Core Recorder.
# Keep these values stable: they are the trigger codes used in EEG analysis.
MARKER_UDP_CODES = {
    "trial_start:left": 1,
    "trial_start:right": 2,
    "trial_start:rest": 3,
    "trial_start:correct:left": 10,
    "trial_start:correct:right": 11,
    "trial_start:error:left": 12,
    "trial_start:error:right": 13,
    "trial_input:correct:left": 14,
    "trial_input:correct:right": 15,
    "trial_input:error:left": 16,
    "trial_input:error:right": 17,
    "trial_end:correct:left": 18,
    "trial_end:correct:right": 19,
    "trial_end:error:left": 20,
    "trial_end:error:right": 21,
}


class LSLMarkerSender:
    def __init__(
        self,
        stream_name="MindTheLasersMarkers",
        stream_type="Markers",
        source_id="mind_the_lasers_markers",
    ):
        # pylsl is imported only when LSL mode is selected.
        from pylsl import StreamInfo, StreamOutlet

        info = StreamInfo(
            name=stream_name,
            type=stream_type,
            channel_count=1,
            nominal_srate=0,
            channel_format="string",
            source_id=source_id,
        )

        self.outlet = StreamOutlet(info)

        print(
            f"Marker transport: lsl; LSL marker stream created: "
            f"name='{stream_name}', type='{stream_type}'"
        )

    def send(self, marker):
        marker = str(marker)

        self.outlet.push_sample([marker])

        print(f"LSL marker sent: {marker}")

    def close(self):
        self.outlet = None


class UDPMarkerSender:
    def __init__(self, udp_ip="127.0.0.1", udp_port=1000):
        self.address = (udp_ip, udp_port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.closed = False
        print(
            f"Marker transport: udp; destination={udp_ip}:{udp_port}; "
            "ASCII trigger codes enabled"
        )

    def send(self, marker):
        if self.closed:
            raise RuntimeError("UDP marker sender is closed")

        marker = str(marker)
        if marker not in MARKER_UDP_CODES:
            raise ValueError(
                f"Unmapped UDP marker '{marker}'. "
                "Add it to MARKER_UDP_CODES in marker_sender.py"
            )

        code = MARKER_UDP_CODES[marker]
        payload = str(code).encode("ascii")
        self.socket.sendto(payload, self.address)
        print(f"UDP marker sent: {marker} -> {code}")

    def close(self):
        if not self.closed:
            self.socket.close()
            self.closed = True


def create_marker_sender(transport="lsl", udp_ip="127.0.0.1", udp_port=1000):
    transport = str(transport).lower()
    if transport == "lsl":
        return LSLMarkerSender()
    if transport == "udp":
        return UDPMarkerSender(udp_ip=udp_ip, udp_port=udp_port)
    raise ValueError(
        f"Invalid MARKER_TRANSPORT '{transport}'. Expected 'lsl' or 'udp'."
    )