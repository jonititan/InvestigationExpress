# Investigation Express
AI² Artificial Intelligence for Accident Investigation

A project started in collaboration with my twin for the OpenCV 2021 Spatial AI Competition.
https://opencv.org/opencv-spatial-ai-competition/

The rational behind applying spatial AI in an Accident Investigation context is discussed in my blog post for the Cranfield University Safety & Accident investigation Centre.
https://saiblog.cranfield.ac.uk/blog/ai-in-accident-investigation

Assuming you have raspbian lite and ansible already setup on the raspberry pi's in your Bramble cluster you are then read to creae an inventory file.
Mine is called Bramble Inventory.
It has a group of the pi's named cameras in it which this playbook works on.

ansible-playbook -i BrambleInventory daiRTSP.yml


## Hints
finding the mac address of your rpi

ip link show dev eth0 | awk ' /link\/ether/ { print $2 }'

changing from password ssh to key ssh
https://www.raspberrypi.org/documentation/remote-access/ssh/passwordless.md

make sure udev rules are set otherwise you get err code 3 when tring to start the OakD
https://docs.luxonis.com/en/latest/pages/troubleshooting/#failed-to-boot-the-device-1-3-ma2480-err-code-3
