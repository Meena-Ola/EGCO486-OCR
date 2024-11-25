
def label_ENG():
    label_encoder_ENG={chr(i+48): i+1 for i in range(10)}  # 0-9 -> 1-10
    # Uppercase letters 'A-Z' will be encoded as 11-36
    label_encoder_ENG.update({chr(i+65): i+11 for i in range(26)})  # A-Z -> 11-36
    # Lowercase letters 'a-z' will be encoded as 37-62
    label_encoder_ENG.update({chr(i+97): i+37 for i in range(26)})  # a-z -> 37-62
    label_encoder_ENG.update({'?':63,"!":64,".":65,",":66,"(":67,")":68,"-":69,"'":70,"\"":71,":":72,";":73})
    return label_encoder_ENG

def label_thai():
    thai_chars = [chr(i) for i in range(0x0E01, 0x0E2E + 1)]
    label_encoder_thai = {char: idx+1  for idx, char in enumerate(thai_chars)}
    return label_encoder_thai

       