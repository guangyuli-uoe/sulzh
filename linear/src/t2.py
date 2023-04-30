

for i in range(0, 3):
    print(i)

w_i = '我爱你m了c'
features = []
# if len(w_i) >= 5:
#     for k in range(len(w_i)):
#         if 0 < k < 5:
#             features.append('14_' + w_i[0:k])

# for k in range(len(w_i)+1):
#     if len(w_i) >= 5:
#         if 0 < k < 5:
#             features.append('14_' + w_i[0:k])
#     elif len(w_i) == 4:
#         # print('here')
#         if k > 0:
#             features.append('14_' + w_i[:k] + str(k))


for k in range(len(w_i)+1):
    if len(w_i) >= 4:
        if 0 < k <= 4:
            # features.append('14_' + w_i[:k])
            features.append('15_' + w_i[-k:])
    else:
        if k > 0:
            # features.append('14_' + w_i[:k])
            features.append('15_' + w_i[-k:])


# for k in range(len(w_i)+1):
#
#     if k > 0:
#         features.append('14_' + w_i[:k])
#     # else:
#     #     if k > 0:
#     #         features.append('14_' + w_i[:k])


print(features)


for i in range(1, 2):
    print(i)
