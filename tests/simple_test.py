


def prediction_t3():
    t3 = 2 * t2 - t1
    return t3



while True:
    print("Enter t1 and t2:")
    txt_input = input().strip()
    if txt_input.lower() == 'exit':
        break
    t1, t2 = map(float, txt_input.split())
    print("Predicted t3:", prediction_t3())


