class HeapSort:
    def __init__(self, A):
        self.A = A.copy()
        self.heap_size = len(self.A)

    def left(self, i): return 2 * i + 1
    def right(self, i): return 2 * i + 2
    def parent(self, i): return (i - 1) // 2


    def max_heapify(self, i): #여기서 ppt랑 다르게 <=로 하면 작동이 안되는데 index가 0부터 시작해서 그런가
        L = self.left(i)
        R = self.right(i)

        if L < self.heap_size and self.A[L]>self.A[i]:
            largest = L
        else:
            largest = i
        
        if R < self.heap_size and self.A[R]>self.A[largest]:
            largest = R

        if largest != i:
            self.A[i], self.A[largest] = self.A[largest], self.A[i]
            self.max_heapify(largest)

    def build_max_heap(self):
        self.heap_size = len(self.A)
        for i in range((self.heap_size // 2) - 1, -1, -1):
            self.max_heapify(i)

    def sort(self):
        self.build_max_heap()
        for i in range(len(self.A) - 1, 0, -1):
            self.A[0], self.A[i] = self.A[i], self.A[0]
            self.heap_size -= 1
            self.max_heapify(0)
        return self.A



class MergeSort:
    def __init__(self, A):
        self.A = A.copy()

    def merge(self, p, q, r):
        n1 = q - p + 1
        n2 = r - q

        L = [self.A[p + i] for i in range(n1)] + [float('inf')]
        R = [self.A[q + 1 + j] for j in range(n2)] + [float('inf')]

        i = j = 0
        for k in range(p, r + 1):
            if L[i] <= R[j]:
                self.A[k] = L[i]
                i += 1
            else:
                self.A[k] = R[j]
                j += 1

    def merge_sort(self, p, r):
        if p < r:
            q = (p + r) // 2
            self.merge_sort(p, q)
            self.merge_sort(q + 1, r)
            self.merge(p, q, r)

    def sort(self):
        self.merge_sort(0, len(self.A) - 1)
        return self.A



class BubbleSort:
    def __init__(self, A):
        self.A = A.copy()

    def sort(self):
        n = len(self.A)
        for i in range(n):
            for j in range(0, n - i - 1):
                if self.A[j] > self.A[j + 1]:
                    self.A[j], self.A[j + 1] = self.A[j + 1], self.A[j]
        return self.A


class InsertionSort:
    def __init__(self, A):
        self.A = A.copy()

    def sort(self):
        for j in range(1, len(self.A)):  #인덱스이슈
            key = self.A[j]
            i = j - 1
            while i >= 0 and self.A[i] > key:
                self.A[i + 1] = self.A[i]
                i -= 1
            self.A[i + 1] = key
        return self.A


class SelectionSort:
    def __init__(self, A):
        self.A = A.copy()

    def sort(self):
        n = len(self.A)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if self.A[j] < self.A[min_idx]:
                    min_idx = j
            self.A[i], self.A[min_idx] = self.A[min_idx], self.A[i]
        return self.A


class QuickSort:
    def __init__(self, A):
        self.A = A.copy()

    def quick_sort(self, arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[0]
        left = [x for x in arr[1:] if x < pivot]
        right = [x for x in arr[1:] if x >= pivot]
        return self.quick_sort(left) + [pivot] + self.quick_sort(right)

    def sort(self):
        self.A = self.quick_sort(self.A)
        return self.A

class LibrarySort:
    def __init__(self, A):
        self.original = A.copy()
        self.A = []  # gap을 포함한 배열

    def binary_search(self, value, S, length):
        low = 0
        high = length
        while low < high:
            mid = (low + high) // 2
            if S[mid] < value:
                low = mid + 1
            else:
                high = mid
        return low

    def rebalance(self, A, begin, end, gap=1):
        r = end
        w = end * 2
        while r >= begin:
            while len(A) <= w + 1:
                A.append(None)
            A[w + 1] = gap
            A[w] = A[r]
            r -= 1
            w -= 2

    def int_log2(self, n):
        count = 0
        while n > 1:
            n = n // 2
            count += 1
        return count

    def sort(self):
        n = len(self.original)
        S = [None] * (2 * n)
        count = 0
        gap = 1

        rounds = self.int_log2(n) + 1  # 라이브러리 없이 log2 구현

        for i in range(rounds):
            step = 1
            for _ in range(i):  # step = 2^i
                step *= 2

            for j in range(step, min(2 * step, n)):
                x = self.original[j]

                # 현재 삽입된 원소만 기준으로 이진 탐색
                active = [val for val in S if val is not None and val != 'gap']
                ins = self.binary_search(x, active, len(active))

                # 실제 삽입 위치 찾기 (갭 고려)
                idx = 0
                placed = 0
                while placed < ins or (S[idx] is not None and S[idx] != 'gap'):
                    if S[idx] is not None and S[idx] != 'gap':
                        placed += 1
                    idx += 1

                while S[idx] is not None:
                    idx += 1
                S[idx] = x
                count += 1

            self.rebalance(S, 0, count - 1)

        self.A = [x for x in S if x is not None and x != 'gap']
        return self.A

class TimSort:
    def __init__(self, A):
        self.A = A.copy()
        self.RUN = 32
        self.merge = MergeSort(self.A)  # merge는 원본 참조하므로 OK

    def sort(self):
        n = len(self.A)

        # 1. RUN 구간 삽입 정렬
        for i in range(0, n, self.RUN):
            j = min(i + self.RUN - 1, n - 1)
            self.A[i : j + 1] = InsertionSort(self.A[i : j + 1]).sort()

        # 2. 병합 단계
        size = self.RUN
        while size < n:
            for left in range(0, n, 2 * size):
                mid = min(n - 1, left + size - 1)
                right = min(n - 1, left + 2 * size - 1)
                if mid < right:
                    self.merge.merge(left, mid, right)
            size *= 2

        return self.A


class CocktailShakerSort:
    def __init__(self, A):
        self.A = A.copy()

    def sort(self):
        n = len(self.A)
        swapped = True

        while swapped:
            swapped = False

            # 왼쪽 → 오른쪽
            for i in range(0, n - 1):
                if self.A[i] > self.A[i + 1]:
                    self.A[i], self.A[i + 1] = self.A[i + 1], self.A[i]
                    swapped = True

            if not swapped:
                break  # 정렬 완료

            swapped = False

            # 오른쪽 → 왼쪽
            for i in range(n - 2, -1, -1):
                if self.A[i] > self.A[i + 1]:
                    self.A[i], self.A[i + 1] = self.A[i + 1], self.A[i]
                    swapped = True

        return self.A


class CombSort:
    def __init__(self, A):
        self.A = A.copy()

    def sort(self):
        gap = len(self.A)
        shrink = 1.3
        sorted_flag = False

        while not sorted_flag:
            gap = int(gap / shrink)

            if gap > 1:
                sorted_flag = False
            else:
                gap = 1
                sorted_flag = True

            i = 0
            while i + gap < len(self.A):
                if self.A[i] > self.A[i + gap]:
                    self.A[i], self.A[i + gap] = self.A[i + gap], self.A[i]
                    sorted_flag = False
                i += 1

        return self.A


class Tree:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []
class TournamentSort:
    def __init__(self, A):
        self.original = A.copy()

    def promote(self, winner, loser):
        return Tree(winner.value, winner.children + [loser])

    def play_game(self, tree1, tree2):
        if tree1.value <= tree2.value:
            return self.promote(tree1, tree2)
        else:
            return self.promote(tree2, tree1)

    def play_round(self, trees):
        if not trees:
            return []
        if len(trees) == 1:
            return [trees[0]]
        next_round = []
        i = 0
        while i + 1 < len(trees):
            next_round.append(self.play_game(trees[i], trees[i + 1]))
            i += 2
        if i < len(trees):
            next_round.append(trees[i]) 
        return next_round

    def play_tournament(self, trees):
        while len(trees) > 1:
            trees = self.play_round(trees)
        return trees[0] 

    def extract_winner(self, tree):
        return tree.value, tree.children

    def sort(self):
        forest = [Tree(x) for x in self.original]
        result = []
        while forest:
            winner_tree = self.play_tournament(forest)
            winner, forest = self.extract_winner(winner_tree)
            result.append(winner)
        return result


class IntroSort:
    def __init__(self, A):
        self.A = A.copy()

    def sort(self):
        maxdepth = self.log2(len(self.A)) * 2
        self.introsort(0, len(self.A) - 1, maxdepth)
        return self.A

    def introsort(self, start, end, depth):
        size = end - start + 1
        if size < 16:
            # ✅ InsertionSort 범위 지정 없이 사용
            sorted_sub = InsertionSort(self.A[start:end + 1]).sort()
            self.A[start:end + 1] = sorted_sub
        elif depth == 0:
            # ✅ HeapSort도 범위 없이 사용 (슬라이스 후 덮어쓰기)
            sorted_sub = HeapSort(self.A[start:end + 1]).sort()
            self.A[start:end + 1] = sorted_sub
        else:
            p = self.partition(start, end)
            self.introsort(start, p - 1, depth - 1)
            self.introsort(p + 1, end, depth - 1)

    def partition(self, low, high):
        pivot = self.A[high]
        i = low
        for j in range(low, high):
            if self.A[j] <= pivot:
                self.A[i], self.A[j] = self.A[j], self.A[i]
                i += 1
        self.A[i], self.A[high] = self.A[high], self.A[i]
        return i

    def log2(self, n):
        res = 0
        while n > 1:
            n //= 2
            res += 1
        return res



if __name__ == "__main__":
    import random
    import time
    import sys

    with open("results.txt", "w") as f:
        sys.stdout = f
        for size in range(3):
            data = random.sample(range(10**(size+3)*2), 10**(size+3))
            for i in range(10):
                alg = MergeSort(data)
                start = time.time()
                alg.sort()
                print(f"MergeSort {10**(size+3)} : ", time.time() - start)

                alg = HeapSort(data)
                start = time.time()
                alg.sort()
                print(f"HeapSort {10**(size+3)} : ", time.time() - start)
                
                alg = BubbleSort(data)
                start = time.time()
                alg.sort()
                print(f"BubbleSort {10**(size+3)} : ", time.time() - start)

                alg = InsertionSort(data)
                start = time.time()
                alg.sort()
                print(f"InsertionSort {10**(size+3)} : ", time.time() - start)

                alg = SelectionSort(data)
                start = time.time()
                alg.sort()
                print(f"SelectionSort {10**(size+3)} : ", time.time() - start)

                alg = QuickSort(data)
                start = time.time()
                alg.sort()
                print(f"QuickSort {10**(size+3)} : ", time.time() - start)

                alg = LibrarySort(data)
                start = time.time()
                alg.sort()
                print(f"LibrarySort {10**(size+3)} : ", time.time() - start)

                alg = TimSort(data)
                start = time.time()
                alg.sort()
                print(f"TimSort {10**(size+3)} : ", time.time() - start)
                
                alg = CocktailShakerSort(data)
                start = time.time()
                alg.sort()
                print(f"CocktailShakerSort {10**(size+3)} : ", time.time() - start)

                alg = CombSort(data)
                start = time.time()
                alg.sort()
                print(f"CombSort {10**(size+3)} : ", time.time() - start)

                alg = TournamentSort(data)
                start = time.time()
                alg.sort()
                print(f"TournamentSort {10**(size+3)} : ", time.time() - start)

                alg = IntroSort(data)
                start = time.time()
                alg.sort()
                print(f"IntroSort {10**(size+3)} : ", time.time() - start)

    with open("sorted_results.txt", "w") as f:
        sys.stdout = f
        for size in range(3):
            data = sorted(random.sample(range(10**(size+3)*2), 10**(size+3)))
            for i in range(10):
                alg = MergeSort(data)
                start = time.time()
                alg.sort()
                print(f"MergeSort {10**(size+3)} : ", time.time() - start)

                alg = HeapSort(data)
                start = time.time()
                alg.sort()
                print(f"HeapSort {10**(size+3)} : ", time.time() - start)
                
                alg = BubbleSort(data)
                start = time.time()
                alg.sort()
                print(f"BubbleSort {10**(size+3)} : ", time.time() - start)

                alg = InsertionSort(data)
                start = time.time()
                alg.sort()
                print(f"InsertionSort {10**(size+3)} : ", time.time() - start)

                alg = SelectionSort(data)
                start = time.time()
                alg.sort()
                print(f"SelectionSort {10**(size+3)} : ", time.time() - start)

                alg = QuickSort(data)
                start = time.time()
                alg.sort()
                print(f"QuickSort {10**(size+3)} : ", time.time() - start)

                alg = LibrarySort(data)
                start = time.time()
                alg.sort()
                print(f"LibrarySort {10**(size+3)} : ", time.time() - start)

                alg = TimSort(data)
                start = time.time()
                alg.sort()
                print(f"TimSort {10**(size+3)} : ", time.time() - start)
                
                alg = CocktailShakerSort(data)
                start = time.time()
                alg.sort()
                print(f"CocktailShakerSort {10**(size+3)} : ", time.time() - start)

                alg = CombSort(data)
                start = time.time()
                alg.sort()
                print(f"CombSort {10**(size+3)} : ", time.time() - start)

                alg = TournamentSort(data)
                start = time.time()
                alg.sort()
                print(f"TournamentSort {10**(size+3)} : ", time.time() - start)

                alg = IntroSort(data)
                start = time.time()
                alg.sort()
                print(f"IntroSort {10**(size+3)} : ", time.time() - start)


    with open("reverse_sorted_results.txt", "w") as f:
        sys.stdout = f
        for size in range(3):
            data = sorted(random.sample(range(10**(size+3)*2), 10**(size+3)), reverse=True)
            for i in range(10):
                alg = MergeSort(data)
                start = time.time()
                alg.sort()
                print(f"MergeSort {10**(size+3)} : ", time.time() - start)

                alg = HeapSort(data)
                start = time.time()
                alg.sort()
                print(f"HeapSort {10**(size+3)} : ", time.time() - start)
                
                alg = BubbleSort(data)
                start = time.time()
                alg.sort()
                print(f"BubbleSort {10**(size+3)} : ", time.time() - start)

                alg = InsertionSort(data)
                start = time.time()
                alg.sort()
                print(f"InsertionSort {10**(size+3)} : ", time.time() - start)

                alg = SelectionSort(data)
                start = time.time()
                alg.sort()
                print(f"SelectionSort {10**(size+3)} : ", time.time() - start)

                alg = QuickSort(data)
                start = time.time()
                alg.sort()
                print(f"QuickSort {10**(size+3)} : ", time.time() - start)

                alg = LibrarySort(data)
                start = time.time()
                alg.sort()
                print(f"LibrarySort {10**(size+3)} : ", time.time() - start)

                alg = TimSort(data)
                start = time.time()
                alg.sort()
                print(f"TimSort {10**(size+3)} : ", time.time() - start)
                
                alg = CocktailShakerSort(data)
                start = time.time()
                alg.sort()
                print(f"CocktailShakerSort {10**(size+3)} : ", time.time() - start)

                alg = CombSort(data)
                start = time.time()
                alg.sort()
                print(f"CombSort {10**(size+3)} : ", time.time() - start)

                alg = TournamentSort(data)
                start = time.time()
                alg.sort()
                print(f"TournamentSort {10**(size+3)} : ", time.time() - start)

                alg = IntroSort(data)
                start = time.time()
                alg.sort()
                print(f"IntroSort {10**(size+3)} : ", time.time() - start)


