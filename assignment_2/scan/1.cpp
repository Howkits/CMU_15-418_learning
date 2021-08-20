#include<iostream>
using namespace std;

void modify(int &n)
{
    n++;
}

int main()
{
    int n=1;
    modify(n);
    cout<<n<<endl;

    return 0;
}