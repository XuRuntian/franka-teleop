import spnav

spnav.open()  # 打开设备
try:
    while True:
        event = spnav.poll_event()
        if event:
            if isinstance(event, spnav.MotionEvent):
                print("Translation:", event.translation)
                print("Rotation:", event.rotation)
            elif isinstance(event, spnav.ButtonEvent):
                print("Button", event.button, "state:", event.press)
except KeyboardInterrupt:
    spnav.close()

