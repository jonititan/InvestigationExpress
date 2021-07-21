# Investigation Express
AI² Artificial Intelligence for Accident Investigation

A project started in collaboration with my twin for the OpenCV 2021 Spatial AI Competition.
https://opencv.org/opencv-spatial-ai-competition/

The rational behind applying spatial AI in an Accident Investigation context is discussed in my blog post for the Cranfield University Safety & Accident investigation Centre.
https://saiblog.cranfield.ac.uk/blog/ai-in-accident-investigation

Assuming you have raspbian lite and ansible already setup on the raspberry pi's in your Bramble cluster you are then read to create an inventory file.
Mine is called BrambleInventory.  https://docs.ansible.com/ansible/latest/user_guide/intro_inventory.html
I put the IP's of the RPi's instead of the URL's
It has a group of the pi's named cameras in it which this playbook works on.

ansible-playbook -i BrambleInventory daiRTSP.yml

Once you have run the playbook it will get to the last task and stay there.  I will change it to a service at some point so it can return task sucessful but for now it just runs the python rtsp example provided by Luxonis.

https://github.com/luxonis/depthai-experiments/tree/master/rtsp-streaming

To view the RTSP stream(or streams) you have created you need to use VLC or similar as discussed in their example.
Just swap localhost for the ip of the stream you want to look at.

## Hints
finding the mac address of your rpi

ip link show dev eth0 | awk ' /link\/ether/ { print $2 }'

changing from password ssh to key ssh
https://www.raspberrypi.org/documentation/remote-access/ssh/passwordless.md

make sure udev rules are set otherwise you get err code 3 when tring to start the OakD
https://docs.luxonis.com/en/latest/pages/troubleshooting/#failed-to-boot-the-device-1-3-ma2480-err-code-3
