# datasets.py

def xor_dataset():
    # inputs are length-2; outputs are length-1
    data = []
    for a in [0.0, 1.0]:
        for b in [0.0, 1.0]:
            x = [a, b]
            y = [1.0 if (a != b) else 0.0]
            data.append((x, y))
    return data

def two_bit_adder_dataset():
    data = []
    bits = [0.0, 1.0]
    for a0 in bits:
        for b0 in bits:
            for c0 in bits:
                for a1 in bits:
                    for b1 in bits:
                        # low bit
                        s = int(a0 + b0 + c0)
                        s0 = float(s % 2)
                        c1 = s // 2
                        # high bit
                        s_hi = int(a1 + b1 + c1)
                        s1 = float(s_hi % 2)
                        c2 = float(s_hi // 2)
                        x = [a0, b0, c0, a1, b1]
                        y = [s0, s1, c2]
                        data.append((x, y))
    return data
