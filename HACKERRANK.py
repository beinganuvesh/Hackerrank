#!/usr/bin/env python
# coding: utf-8

# In[110]:


import bisect
arr=[1,2,3,4,5,6,7,9]
bisect.bisect_right(arr, 8)


# In[111]:


n=7
arr=[1,2,3,4,5,6,7]
m=2
prev_sum=[0 for i in range(n+1)]
for i in range(1, n+1):
    prev_sum[i]=prev_sum[i-1]+arr[i-1]
prev_sum


# In[112]:


idx=bisect.bisect_left(arr, m)
idx, prev_sum[idx]


# In[ ]:





# In[ ]:





# In[13]:


import heapq
class MedianFinder:
  
    def __init__(self):
        self.left=[]
        self.right=[]
    
    def addNum(self, num):
      if len(self.left)==0:
        heapq.heappush(self.left, -num)
      elif num<=self.left[0]:
        heapq.heappush(self.left, -num)
      else:
        heapq.heappush(self.right, num)
        
      #Check if the sizes of the queues are more than 1!
      if len(self.left)-len(self.right)>1:
          heapq.heappush(self.right, -heapq.heappop(self.left))
      elif len(self.right)-len(self.left)>1:
          heapq.heappush(self.left, -heapq.heappop(self.right))
          

    def findMedian(self):
        if len(self.left)>len(self.right):
          return -self.left[0]
        elif len(self.left)<len(self.right):
          return self.right[0]
        else:
          return (self.right[0]-self.left[0])/2


# In[14]:


obj=MedianFinder()


# In[15]:


obj.addNum(1)
obj.addNum(2)


# In[17]:


obj.findMedian()


# In[ ]:





# In[ ]:





# In[1]:


import heapq as pq
pq.heappush()


# In[23]:


A=5
D=[14,12,10,13,14]
E=[1,0,2,3,4]
F=[22,24,21,20,24]


# In[24]:


l=[]
for i in range(A):
    new_row=[D[i], E[i], F[i]]
    l.append(new_row)
l


# In[25]:


arr=sorted(l, key=lambda l:l[2])
arr


# In[ ]:





# In[ ]:





# In[27]:


def solve(A, B, C, D, E, F):
    l=[]
    for i in range(A):
        #if D[i]>=B and E[i]<=C:
            new_row=[D[i], E[i], F[i]]
            l.append(new_row)
    arr=sorted(l, key=lambda l:l[2])
    res=[]
    days=0
    cost=0
    for i in range(2):
        days+=arr[i][1]-arr[i][0]+1
        cost+=arr[i][2]
    return days


# In[28]:


solve(A, 0, 0, D, E, F)


# In[ ]:





# In[ ]:





# In[15]:


def threeSum(nums):
    lst=[]
    l = len(nums)
    nums.sort()
    for i in range(l-1):
        r=pairSum(nums[i+1:], -nums[i])
        for p in r:
            lst.append([nums[i], p[0], p[1]])
    return lst
                  
def pairSum(arr, x):
    result=[]
    i=0
    j=len(arr)-1
    while i<=j:
        s = arr[i]+arr[j]
        if s>x:
            j-=1
        elif s<x:
            i-=1
        else:
            if arr[i]==arr[j]:
                p = (arr.count(arr[i]-1))*(arr.count(arr[i]))//2
            else:
                p = (arr.count(arr[i]))*(arr.count(arr[j]))
                
            for t in range(p):
                result.append([arr[i], arr[j]])
            i+=arr.count(arr[i])
            j-=arr.count(arr[j])
            
    return result


# In[ ]:


nums = [-1,0,1,2,-1,-4]
threeSum(nums)


# In[ ]:





# In[ ]:





# In[1]:


import sys
def runningTime(arr):
    l = len(arr)
    for i in range(l-1):
        mini = sys.maxsize
        for j in range(i+1, l):
            if arr[i]>arr[j]:
                temp = arr[j]
                if temp<mini:
                    mini = temp
                    idx = j
                mini = min(temp, mini)
        arr[i], arr[j] = arr[j], arr[i]
        
    return arr


# In[7]:


arr = [4,3,2,5,1]
runningTime(arr)


# In[ ]:





# In[ ]:





# In[192]:


def Coins_DP(value, c):
    n = len(c)
    dp = [[0 for j in range(value+1)]for i in range(n+1)]
    #Initailization of the dp array.
    for i in range(n): 
        dp[i][0] = 1
    
    for i in range(1, n+1):
        for j in range(1, value+1):
            
            if j<c[i-1]:
                dp[i][j] = dp[i-1][j]
            else:
                a1 = dp[i][j-c[i-1]]
                a2 = dp[i-1][j]
                dp[i][j] = (a1+a2)
                
    return dp[n][value]


# In[193]:


c = [2,5,3,6]
v = 10
Coins_DP(v, c)


# In[ ]:





# In[ ]:





# In[86]:


def Knap(arr, k, i, n):
    if i>=n:
        return 0
    if arr[i]<=k:
        smallerOutput1 = arr[i] + Knap(arr, k-arr[i], i, n)
        smallerOutput2 = Knap(arr, k, i+1, n)
        ans = max(smallerOutput1, smallerOutput2)
    else:
        ans = Knap(arr, k, i+1, n)
    return ans


# In[ ]:





# In[ ]:





# In[95]:


def Knap_Dp(arr, n, k):
    dp = [[0 for j in range(k+1)] for i in range(n+1)]
    
    for i in range(1, n+1):
        for j in range(1, k+1):
            if j<arr[i-1]:
                dp[i][j] = dp[i-1][j]
            else:
                a1 = arr[i-1] + dp[i][j-arr[i-1]]
                a2 = dp[i-1][j]
                dp[i][j] = max(a1, a2)
                
    return dp[n][k]


# In[96]:


Knap_Dp(a, 3, k)


# In[ ]:





# In[ ]:





# In[43]:


def maxMin(k, arr):
    l = len(arr)
    ar = sorted(arr)
    ans = sys.maxsize

    for i in range(l-k+1):
        sub = ar[i:i+k]
        x = sub[-1]
        y = sub[0]
        unfairness = x-y
        ans = min(ans, unfairness)
    return ans


# In[41]:


k = 3
a = [3, 10, 100, 300, 200, 1000 , 20 ,30]


# In[ ]:





# In[ ]:





# In[26]:


import queue
class Graph:
    def __init__(self, nvertices):
        self.nvertices = nvertices
        self.adjMatrix = [[0 for j in range(nvertices)] for i in range(nvertices)]
        
    def add_edge(self, v1, v2):
        self.adjMatrix[v1][v2]=1
        self.adjMatrix[v2][v1]=1
    

    def BFS_Helper(self, sv, visited):
        num=0
        q = queue.Queue()
        q.put(sv)
        visited[sv]=True
        num+=1
        
        while q.empty() is False:
            current = q.get()
            for i in range(self.nvertices):
                if self.adjMatrix[current][i]==1 and visited[i]==False:
                    visited[i]=True
                    q.put(i)
                    num+=1
        return num

    def BFS(self):
        result = []
        visited = [False for i in range(self.nvertices)]
        for i in range(self.nvertices):
            if visited[i]!=True:
                c=0
                c = self.BFS_Helper(i, visited)
                result.append(c)

        return result


# In[29]:


def journeyToMoon(n, astronaut):
    g = Graph(n)
    #Connecting the edges
    for i in astronaut:
        g.add_edge(i[0], i[1])

    r = g.BFS()
    #Now counting the disjoint set.
    ans=0
    a = r[0]
    for i in range(1, len(r)):
        ans = ans + a*r[i]
        a = a+r[i]

    return ans


# In[30]:


n = 5
a = [[0,1],[2,3],[0,4]]
journeyToMoon(n, a)


# In[ ]:





# In[ ]:





# In[10]:


def Operator(arr):
    #Base Case.
    if len(arr)==1:
        s = str(arr[0])
        return s
        
    smallerOutput = Operator(arr[1:])
    
    a1 = arr[0] + int(smallerOutput)
    if a1%101==0:
        return str(arr[0]) + "+" + smallerOutput
    a2 = arr[0] - int(smallerOutput)
    if a2%101==0:
        return str(arr[0]) + "-" + smallerOutput
    a3 = arr[0] * int(smallerOutput)
    if a3%101==0:
        return str(arr[0]) + "*" + smallerOutput
        
    


# In[ ]:





# In[ ]:





# In[4]:


def SumOfDigits(num):
    #Base Case
    if num<10:
        return num
    
    smallerOutput = SumOfDigits(num//10)
    if (smallerOutput+num%10)<10:
        return smallerOutput+num%10
    else:
        return SumOfDigits(smallerOutput+num%10)


# In[5]:


SumOfDigits(148)


# In[ ]:





# In[ ]:





# In[50]:


def encryption(s):
    s = s.strip()
    l = len(s)
    r = math.floor(math.sqrt(l))
    c = math.ceil(math.sqrt(l))
    if r*c < l:
        r = max(r,c)
        c = max(r,c)
    
    output=[]
    i=0
    while len(l)>i:
        string = s[i:i+c]
        output.append(string)
        i+=c
        
    return output


# In[ ]:





# In[ ]:





# In[43]:


def acmTeam(topic):
    l = len(topic)
    m=-1
    number=0

    for i in range(l-1):
        s1 = topic[i]
        for j in range(i+1, l):
            s2 = topic[j]
            c=0
            for k in range(len(s1)):
                if s1[k]=='1' or s2[k]=='1':
                    c+=1
                m = max(m, c)

    return m


# In[36]:


t[0]


# In[44]:


t = ['10101', '11100', '11010', '00101']
acmTeam(t)


# In[ ]:





# In[ ]:





# In[9]:


def Birds(arr):
    d={}
    for i in arr:
        d[i]=d.get(i,0)+1
    maxi=-1
    t=0
    for i in sorted(arr):
        if d[i]>maxi:
            maxi=d[i]
            t=i
    return t, maxi
    


# In[10]:


Birds(arr)


# In[ ]:





# In[ ]:





# In[25]:


def Binary_Search(arr, start, end, target):
  if start>end:
    return -1
  
  mid = (start+end)//2
  
  if arr[mid]==target:
    return mid
  elif arr[mid]<target:
    return Binary_Search(arr, mid+1, end, target)
  else:
    return Binary_Search(arr, start, mid-1, target)


# In[26]:


l = [1,3,5,6]
t = 2


# In[27]:


n = len(l)-1


# In[28]:


Binary_Search(l, 0, n, t)


# In[ ]:





# In[ ]:





# In[72]:


def subsets(arr):
    if len(arr)==0:
        return [[]]
    smallerOutput = subsets(arr[1:])
    
    ans=[]
    for i in smallerOutput:
        if i not in ans:
            ans.append(i)
        
    for i in smallerOutput:
        if (sorted([arr[0]]+i) not in ans):
            ans.append(sorted([arr[0]]+i))
    return ans


# In[73]:


subsets([2,1,2,1,3])


# In[ ]:





# In[ ]:





# In[67]:


def Place_Flowers(arr, n):
    c=0
    l = len(arr)
    for i in range(1, l, 2):
        if arr[i]==0 and (arr[i-1]==0 or i==0) and (arr[i+1]==0 or i==l-1):
            arr[i-1]=1
            c+=1
    if c==n:
        return True
    else:
        return False


# In[66]:


l = [1,0,0,0,1]
n = 2
Place_Flowers(l, nb)


# In[ ]:





# In[ ]:





# # Format of HackerRank Input

# In[ ]:


import math
import os
import random
import re
import sys
from functools import reduce

def lcm(a, b):
    return (a*b)//gcd(a, b)

def lcm_list(lst):
    return reduce(lcm, lst)

def gcd(a, b):
    while a % b != 0:
        a, b = b, (a % b)
    return b
def gcd_list(lst):
    return reduce(gcd, lst)

def evenly_distributed(number, divisor):
    return (number%divisor)==0

def get_input():
    first_multiple_input = input().rstrip().split()
    
    n = int(first_multiple_input[0])
    m = int(first_multiple_input[1])
    arr = list(map(int, input().rstrip().split()))
    brr = list(map(int, input().rstrip().split()))
    
    return n, m, arr, brr

def main():
    n, m, a, b = get_input()

    lcm_value = lcm_list(a)

    gcd_value = gcd_list(b)

    c=0
    i = lcm_value

    while i<=gcd_value:
        if evenly_distributed(gcd_value, i):
            c+=1
        i+=lcm_value

    print(c)

if __name__ == "__main__":
    main()


# In[25]:


x = 'beabeefeab'
f = list(set(x))
f


# In[22]:


def validate(s):
    for i in range(len(s)-1):
        if s[i]==s[i+1]:
            return False
    return True

def TwoCharacter(string):
    s = list(set(string))
    max_len = -1
    for x in range(len(s)):
        for y in range(x+1, len(s)):
            new = [ ch for ch in string if ch==s[x] or ch==s[y] ]
            if validate(new):
                max_len = max(max_len, len(new))
    return max_len


# In[23]:


TwoCharacter(x)


# In[ ]:





# In[ ]:





# In[129]:


import queue
class Graph:
    def __init__(self, nvertices):
        self.nvertices = nvertices
        self.adjMat=[[0 for j in range( (self.nvertices)+1)] for i in range( (self.nvertices)+1)]

    def addEdge(self, v1, v2):
        self.adjMat[v1][v2]=1
        self.adjMat[v2][v1]=1

        
    def DFS_Helper(self, sv, visited, vertex):
        q = queue.Queue()
        visited[sv]=True
        q.put(sv)
        vertex+=1
        while q.empty() is False:
            current = q.get()
            for i in range(1, self.nvertices+1):
                if self.adjMat[current][i]==1 and visited[i]==False:
                    visited[i]=True
                    q.put(i)
                    vertex+=1
        return vertex
    def DFS(self, n, c_lib, c_road):
        visited = [False for i in range( self.nvertices+1)]
        cost = 0
        for i in range(1, self.nvertices+1):
            vertex = 0
            if visited[i]==False:
                x = self.DFS_Helper(i, visited, vertex)
                cost = cost + (c_lib) + c_road*(x-1)
        return cost


def roadsAndLibraries(n, c_lib, c_road, cities):
    g = Graph(n)
    #Connecting the edges of graph.
    for i in range(len(cities)):
        g.addEdge(cities[i][0], cities[i][1])

    if c_lib>c_road:
        ans = g.DFS(n, c_lib, c_road)
    else:
        ans = n*c_lib

    return ans


# In[138]:


class Graph:
    def __init__(self, nvertices):
        self.nvertices = nvertices
        self.adjMat=[[0 for j in range( (self.nvertices)+1)] for i in range( (self.nvertices)+1)]

    def addEdge(self, v1, v2):
        self.adjMat[v1][v2]=1
        self.adjMat[v2][v1]=1

        
    def DFS_Helper(self, sv, visited):
        visited[sv]=True
        for i in range( self.nvertices+1):
            if self.adjMat[sv][i]==1 and visited[i]==False:
                self.DFS_Helper(i, visited)

    def DFS(self, n, c_lib, c_road):
        visited = [False for i in range(self.nvertices+1)]
        component=0
        for i in range(1, self.nvertices+1):
            if visited[i]==False:
                self.DFS_Helper(i, visited)
                component+=1
        return component


def roadsAndLibraries(n, c_lib, c_road, cities):
    g = Graph(n)
    #Connecting the edges of graph.
    for i in range(len(cities)):
        g.addEdge(cities[i][0], cities[i][1])

    if c_lib>c_road:
        ans = g.DFS(n, c_lib, c_road)
        return c_road*(n-ans) + (ans*c_lib) 
    else:
        ans = n*c_lib
        return ans
    


# In[139]:


n = 5
c_lib = 6
c_road = 1
cities = [[1,2],
         [2,3],
         [4,1]]


# In[ ]:





# In[ ]:





# In[10]:


import math, sys
def powerSum(X):
    ans = sys.maxsize
    if X==0:
        return 0
    root=int(math.sqrt(X))
    for i in range(1, root+1):
        currentAns = 1+powerSum(X - (i*i))
        ans = min(currentAns, ans)
    return ans
        


# In[15]:


powerSum(25)


# In[ ]:





# In[ ]:





# In[33]:


import numpy as np
def lilysHomework(arr):
    s = sorted(arr)
    d={}
    c=0
    #Storing the correct index if every element.
    for i in range(len(s)):
        d[arr[i]]=i
        
    for i in range(len(arr)):
        if arr[i]!=s[i]:
            t = d[s[i]]
            arr[t], arr[i] = arr[i], arr[t]
            c+=1
    return c


# In[40]:


def solution(a):
    m = {}
    for i in range(len(a)):
        m[a[i]] = i 
        
    sorted_a = sorted(a)
    ret = 0
    for i in range(len(a)):
        if a[i] != sorted_a[i]:
            ret +=1
            ind_to_swap = m[ sorted_a[i] ]
            m[ a[i] ] = m[ sorted_a[i]]
            a[i],a[ind_to_swap] = sorted_a[i],a[i]
    return ret


# In[1]:


ord('A'), chr(91)


# In[ ]:





# In[ ]:





# In[29]:


def squares(a, b):
    c=0
    for i in range(a, b):
        x = math.sqrt(i)
        y = math.floor(x)
        if y-x==0:
            c+=1
    return c


# In[ ]:





# In[ ]:





# In[ ]:


def queensAttack(n, k, c_q, r_q , obstacles):
    for i in range(k):
        QueenAttackHelper(n, start, end, obstacles[k])
    


# In[1]:


def QueenAttackHelper(n, k, i, j, obstacles, c):
    #Edge case
    if i>n or j>n or i<1 or j<1:
        return 
    #Base Case
    if [i,j] in obstacles:
        return 
    else:
        c+=1
    
    #Hypothesis
    QueenAttackHelper(n, k, i+1, j, obstacles,c)
    QueenAttackHelper(n, k, i, j+1, obstacles,c)
    QueenAttackHelper(n, k, i-1, j, obstacles,c)
    QueenAttackHelper(n, k, i, j-1, obstacles,c)
    QueenAttackHelper(n, k, i+1, j+1, obstacles,c)
    QueenAttackHelper(n, k, i-1, j-1, obstacles,c)
    QueenAttackHelper(n, k, i+1, j-1, obstacles,c)
    QueenAttackHelper(n, k, i-1, j+1, obstacles,c)
    
    return c


# In[2]:


o = [[2,3], [4,2], [5,5]]


# In[ ]:





# In[ ]:





# # HARD AND MEDIUM

# In[ ]:





# In[ ]:





# In[18]:


def highestValuePalindrome(s, n, k):
    #Making the string palindrone
    changed=[False for i in range(n)]
    l=list(s)
    i=0
    j=n-1
    
    while i<j:
        if l[i]!=l[j]:
            l[i]=str(max(int(l[i]), int(l[j])))
            l[j]=str(max(int(l[i]), int(l[j])))
            changed[i]=True
            k-=1
        i+=1
        j-=1
         
    if k<0:
        return -1
    if k==0:
        return "".join(l)
        
             
    i=0
    j=n-1
    while k>0 and i<j:
        if l[i]!='9':
            if changed[i]:
                k+=1
            if k>1:
                l[i]='9'
                l[j]='9'
                k-=2
        i+=1
        j-=1
        
    if len(s)%2==1 and k>0:
        l[len(l)//2]='9'
        
    return "".join(l)


# In[21]:


n=4
s='0011'
k=1
highestValuePalindrome(s,n,k)


# In[ ]:





# In[ ]:





# In[108]:


import operator
def sherlockAndAnagrams(s):
    l=len(s)
    d={}
    for i in range(0, l):
        for j in range(i, l):
            x=sorted(s[i:j+1])
            key="".join(x)
            d[key]=d.get(key,0)+1
    
    cnt=0
    for s in d:
        cnt+=(d[s]*(d[s]-1))//2
        
    return cnt     


# In[109]:


s='cdcd'
sherlockAndAnagrams(s)


# In[ ]:





# In[ ]:





# In[62]:


def pylons(k, arr):
    n=len(arr)
    plants=0
    i=0
    
    while i<n:
        found=False
        for j in range(i+k-1, i-k, -1):
            if j>=0 and j<n and arr[j]==1:
                plants+=1
                i=j+k
                found=True
                break
        if found==False:
            return -1
        
    return plants


# In[63]:


n=7 
k=2
arr=[0, 1, 0, 0, 0, 1, 0]
pylons(k,arr)


# In[ ]:





# In[ ]:





# In[95]:


def sortSentence(s):
        l=s.split()
        v=[]
        for i in range(len(l)):
            v.append((l[i][0:-1], l[i][-1]))
            
        r=sorted(v, key=lambda x:x[1])
        
        ans=""
        for i in range(len(r)):
            ans+=r[i][0]+" "
        return ans.strip()


# In[96]:


s = "is2 sentence4 This1 a3"
sortSentence(s)


# In[ ]:





# In[ ]:





# In[114]:


def rotateTheBox(box):
        n=len(box)
        m=len(box[0])
        
        temp=[[box[j][i] for j in range(n)]for i in range(m)]
        for i in range(len(temp)):
            temp[i]=temp[i][::-1]
            
        row=len(temp)
        col=len(temp[0])
        i=row-1
        j=col-1
        
        while j>=0:
            i=row-1
            while i>0:
                if temp[i][j]=='.' and temp[i-1][j]=='#':
                    temp[i][j], temp[i-1][j] = temp[i-1][j], temp[i][j]
                i-=1
            j-=1
            
        return temp
        


# In[ ]:





# In[ ]:





# In[237]:


#Expert
def reverseShuffleMerge(s):
    l=len(s)
    unused={}
    for ch in s:
        unused[ch]=unused.get(ch,0)+1
    required={}
    for ch in s:
        required[ch]=unused[ch]//2
     
    ans=[]
    ans.append(s[-1])
    required[s[-1]]-=1
    unused[s[-1]]-=1
        
    for i in range(l-2, -1, -1):
        ch=s[i]
        if required[ch]<=0:
            continue
        else:
            if ans[-1]<ch:
                ans.append(ch)
                required[ch]-=1
                unused[ch]-=1
            else:
                while ans and ans[-1]>ch and unused[ans[-1]]>0:
                    removed=ans.pop(-1)
                    required[removed]+=1
                    unused[removed]-=1
                ans.append(ch)
                required[ch]-=1
                unused[ch]-=1
                
    return "".join(ans)


# In[235]:


s='eggegg'
reverseShuffleMerge(s)


# In[236]:


s='abcdefgabcdefg'
reverseShuffleMerge(s)


# In[ ]:





# In[ ]:





# In[ ]:




