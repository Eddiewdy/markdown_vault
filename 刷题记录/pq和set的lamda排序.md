```c++
priority_queue<pair<int,int>,vector<pair<int,int>>,decltype(cmp) > q(cmp);
```

```c++
set<pair<int, int> ,decltype(cmp)> pq(cmp);
```

```c++
auto cmp = [](const pair<int, int>& a, const pair<int, int>& b) { return a.first < b.first; };
```

上面这种写法当a.first == b.first的时候报错:

![image-20220923222519967](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20220923222519967.png)

```c++
auto cmp = [](const pair<int, int>& a, const pair<int, int>& b) { return a.first == b.first ? (a.second < b.second) : a.first < b.first; };
```

