# Vendored third-party code

Files here are copied **verbatim** from upstream and must stay that way, so
that "this is the published model" is a checkable claim rather than an
assertion. They are excluded from black and ruff for exactly that reason —
reformatting them would silently break the comparison.

| file | upstream | why vendored |
| --- | --- | --- |
| `acrnet_model.py` | [banma12956/AcrNET](https://github.com/banma12956/AcrNET) `model.py` | AcrNET ships no installable package. The architecture must match the checkpoint exactly; a reimplementation that merely looked right would load the weights and produce wrong numbers. |

`tests/test_parity.py` pins the SHA-256 of each file, so drift fails the
suite offline rather than needing a network fetch.
