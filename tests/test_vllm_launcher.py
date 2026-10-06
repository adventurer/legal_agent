import unittest
from pathlib import Path
from unittest.mock import patch

from server import vllm_launcher


class VllmLauncherTests(unittest.TestCase):
    def test_qwen35_awq_checkpoint_uses_twelve_thousand_token_limit(self):
        with (
            patch.object(
                vllm_launcher,
                "ensure_model_weights",
                return_value=Path("/models/Qwen3.5-4B-AWQ"),
            ),
            patch.object(vllm_launcher.subprocess, "Popen") as popen,
        ):
            vllm_launcher.launch_vllm_server(
                "qwen3.5-4b",
                enable_fp8_kv=False,
            )

        command = popen.call_args.args[0]
        self.assertEqual(
            command[command.index("--quantization") + 1],
            "awq",
        )
        self.assertIn("--language-model-only", command)
        reasoning_parser_index = command.index("--reasoning-parser")
        self.assertEqual(command[reasoning_parser_index + 1], "qwen3")
        tool_call_parser_index = command.index("--tool-call-parser")
        self.assertEqual(command[tool_call_parser_index + 1], "qwen3_coder")
        self.assertEqual(command[command.index("--max-model-len") + 1], "12000")
        self.assertEqual(
            command[command.index("--gpu-memory-utilization") + 1],
            "0.75",
        )

    def test_awq_checkpoint_keeps_awq_quantization(self):
        with (
            patch.object(
                vllm_launcher,
                "ensure_model_weights",
                return_value=Path("/models/Qwen2.5-7B-Instruct-AWQ"),
            ),
            patch.object(vllm_launcher.subprocess, "Popen") as popen,
        ):
            vllm_launcher.launch_vllm_server(
                "qwen2.5-7b-awq",
                enable_fp8_kv=False,
            )

        command = popen.call_args.args[0]
        quantization_index = command.index("--quantization")
        self.assertEqual(command[quantization_index + 1], "awq")


if __name__ == "__main__":
    unittest.main()
