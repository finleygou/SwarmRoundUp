import numpy as np

# angle_list = [6,3,4,5,2,0,7]
# print(np.sort(angle_list)[0])

flag = [True, True, True, True,True, True]

if all(flag) == False:
    print('collide!!!!!!!')
    r_l = -20
else:
    print('safe')