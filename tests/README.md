# Testing scripts

To check the efficiency of your model, you can use these scripts:

### model_depth_account_test.py

This test checks if the model considers depth into account at all. Use this to check if the fourth channel gets fed into the model and has any significant mathematical value.
The test works by wiping the depth data, and then seeing if there is any difference in the results.

### model_no_rgb_test.py

This test checks if the model is not lazy and can use depth for detection. Use this to test the model's performance in limited visibility.
This test works by wiping the rgb data, and then seeing if the model can still predict a bounding box reliably.

### Usage

Please ensure that you've fitted the directories defined in the beginning of a script before running it. The tests are conducted automatically on the provided model.
