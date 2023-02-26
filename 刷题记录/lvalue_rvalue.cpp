#include<cstdio>
#include <utility>
using namespace std;

class Base{
public:
      Base(const Base& b){
            // copy construct
            printf("copy construct");
      }

      Base(Base&& b){
            //move construct
            printf("move construct");
      }

      Base(){
            // default construct
      }
};

template<typename T>
void ProxyFn(T&& arg){
      Base(std::forward<T>(arg));
}

template <class _Tp>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR _Tp&&
forward(typename remove_reference<_Tp>::type& __t) _NOEXCEPT {
  return static_cast<_Tp&&>(__t);
}

int main(){
     Base b;
     ProxyFn(b);
}