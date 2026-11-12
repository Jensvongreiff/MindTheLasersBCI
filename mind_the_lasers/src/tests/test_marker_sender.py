import sys
import types
import unittest
from unittest.mock import Mock, patch

from mind_the_lasers.src.stream.marker_sender import (
    LSLMarkerSender,
    UDPMarkerSender,
    create_marker_sender,
)


class MarkerSenderTests(unittest.TestCase):
    def test_lsl_sender_preserves_string_marker(self):
        fake_pylsl = types.ModuleType("pylsl")
        fake_pylsl.StreamInfo = Mock(return_value=object())
        fake_outlet = Mock()
        fake_pylsl.StreamOutlet = Mock(return_value=fake_outlet)

        with patch.dict(sys.modules, {"pylsl": fake_pylsl}):
            sender = LSLMarkerSender()
            sender.send("trial_start:correct:left")

        fake_outlet.push_sample.assert_called_once_with(
            ["trial_start:correct:left"]
        )

    def test_udp_sends_ascii_training_and_errp_codes(self):
        fake_socket = Mock()
        with patch("mind_the_lasers.src.stream.marker_sender.socket.socket", return_value=fake_socket):
            sender = UDPMarkerSender("127.0.0.1", 1000)

            sender.send("trial_start:left")
            sender.send("trial_start:rest")
            sender.send("trial_input:error:right")
            sender.send("trial_end:correct:left")

        calls = fake_socket.sendto.call_args_list
        self.assertEqual(calls[0].args, (b"1", ("127.0.0.1", 1000)))
        self.assertEqual(calls[1].args, (b"3", ("127.0.0.1", 1000)))
        self.assertEqual(calls[2].args, (b"17", ("127.0.0.1", 1000)))
        self.assertEqual(calls[3].args, (b"18", ("127.0.0.1", 1000)))

    def test_udp_mode_does_not_import_pylsl(self):
        fake_socket = Mock()
        with patch.dict(sys.modules, {"pylsl": None}), patch(
            "mind_the_lasers.src.stream.marker_sender.socket.socket",
            return_value=fake_socket,
        ):
            sender = create_marker_sender("udp", "127.0.0.1", 1000)

        self.assertIsInstance(sender, UDPMarkerSender)

    def test_invalid_transport_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "lsl.*udp"):
            create_marker_sender("serial")

    def test_unmapped_marker_is_rejected(self):
        with patch("mind_the_lasers.src.stream.marker_sender.socket.socket", return_value=Mock()) as socket_factory:
            sender = UDPMarkerSender()
            socket_factory.assert_called_once()
            with self.assertRaisesRegex(ValueError, "Unmapped UDP marker"):
                sender.send("unknown_marker")

    def test_udp_socket_is_closed(self):
        fake_socket = Mock()
        with patch("mind_the_lasers.src.stream.marker_sender.socket.socket", return_value=fake_socket):
            sender = UDPMarkerSender()
            sender.close()
            sender.close()

        fake_socket.close.assert_called_once_with()
        self.assertTrue(sender.closed)


if __name__ == "__main__":
    unittest.main()
