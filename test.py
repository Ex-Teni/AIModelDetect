import paddle
print(paddle.device.get_device())   # => Should be 'gpu:0'
print(paddle.utils.run_check())     # type: ignore # => Should pass GPU test