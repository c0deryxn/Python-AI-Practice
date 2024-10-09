import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, # 도미 길이 특성 데이터 준비
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, # 도미 무게 특성 데이터 준비
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0] # 빙어 길이 특성 데이터 준비
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9] # 빙어 무게 특성 데이터 준비

length = bream_length + smelt_length # 통합 길이 데이터 준비
weight = bream_weight + smelt_weight # 통합 무게 데이터 준비

fish_data = [[l, w] for l, w in zip(length, weight)] # zip()함수로 각 리스트에서 하나씩 가져와 반환
fish_target = [1] * 35 + [0] * 14 # 도미 1, 빙어 0으로 정의

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target) # 알고리즘 훈련 진행
# print(kn.predict([[35, 400], [0, 0]])) # Test 01 Result: [1 0]
# print(kn.score(fish_data, fish_target)) # Test 02 Result: 1.0

# kn49 = KNeighborsClassifier(n_neighbors=49) # Test 03 Result: 5/7 = 0.714285...
# kn49.fit(fish_data, fish_target)
# print(kn49.score(fish_data, fish_target))

kn5 = KNeighborsClassifier(n_neighbors=5)
kn5.fit(fish_data, fish_target)
print(kn5.score(fish_data, fish_target))