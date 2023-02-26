### 题目4

![image-20220925142651022](https://wangyidipicgo.oss-cn-hangzhou.aliyuncs.com/image-20220925142651022.png)

并查集，按照val的值从低到高，将**相连且val比当前小的节点**并到一起的节点并且并到一起，每次统一同一个val值之间的多个并集的大小，用C(n,2)得到n个选2个的path个数，最后加上n（一个点单独的path）

```c++
struct Union {
    vector<int> f;
    int n;
    Union(int n) : n(n), f(n) { iota(f.begin(), f.end(), 0); }
    bool merge(int x, int y) {
        x = find(x); y = find(y);
        if (x == y) return false;
        f[x] = y;
        return true;
    }
    int find(int x) { return f[x] == x ? x : f[x] = find(f[x]); }
};

class Solution {
public:
    int numberOfGoodPaths(vector<int>& vals, vector<vector<int>>& edges) {
        int n = vals.size();
        Union u(n);
      	// val，同一个val的节点
        map<int, vector<int> > mp;
      	// 图信息
        vector<vector<int> > g(n);
        for(int i = 0; i < n; i++) {
            mp[vals[i]].push_back(i);
        }
        for(auto& edge : edges) {
            g[edge[0]].emplace_back(edge[1]);
            g[edge[1]].emplace_back(edge[0]);
        }
        int res = 0;
        for(auto& [k, v] : mp) {
            for(auto& x : v) {
                for(auto& y : g[x]) {
                    if(vals[x] >= vals[y]) {
                        u.merge(x, y);
                    }
                }
            }
            map<int, int> count;
            for(auto& x : v) {
                count[u.find(x)]++;
            }
            for(auto&[kk, vv] : count) {
                res += vv * (vv - 1) / 2;
            }
            
        }
        return res + n;
    }
};

```

