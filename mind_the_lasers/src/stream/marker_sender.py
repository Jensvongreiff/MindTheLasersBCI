from pylsl import StreamInfo, StreamOutlet


class LSLMarkerSender:
    def __init__(
        self,
        stream_name="MindTheLasersMarkers",
        stream_type="Markers",
        source_id="mind_the_lasers_markers",
    ):
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
            f"LSL marker stream created: "
            f"name='{stream_name}', type='{stream_type}'"
        )

    def send(self, marker):
        marker = str(marker)

        self.outlet.push_sample([marker])

        print(f"LSL marker sent: {marker}")