

def prediction_t3(t1, t2):
    t3 = 2 * t2 - t1
    return t3


while True:
    print("Enter t_touch and t_land:")
    txt_input = input().strip()
    if txt_input.lower() == 'exit':
        break
    t_touch, t_land = map(float, txt_input.split())
    print("Predicted t_shutdown:", prediction_t3(t_touch, t_land))



