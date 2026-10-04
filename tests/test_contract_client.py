import unittest
from unittest.mock import patch

import httpx

from web.contract_client import rewrite_contract


class ContractClientTests(unittest.TestCase):
    def test_includes_server_validation_detail(self):
        request = httpx.Request(
            "POST",
            "http://127.0.0.1:9000/api/v1/contract/rewrite",
        )
        response = httpx.Response(
            422,
            json={"detail": "模型两次修订均遗漏或重排了条款 8 的子条款"},
            request=request,
        )

        with patch("web.contract_client.httpx.post", return_value=response):
            with self.assertRaisesRegex(
                httpx.HTTPStatusError,
                "条款 8 的子条款",
            ):
                rewrite_contract(
                    "http://127.0.0.1:9000",
                    [],
                    "",
                    [],
                )


if __name__ == "__main__":
    unittest.main()
