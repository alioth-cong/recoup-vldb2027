#include "detector_fixed.hpp"
#include "corrected_state.hpp"
#include "closure.hpp"
#include <iostream>
#include <random>
#include <cassert>
using namespace dylo;using namespace dylo_corrected;
struct Tree {std::vector<int> parent;std::vector<double> weight;std::vector<Key> nodeKey;std::vector<std::unordered_map<int,int>> ids;};
void require(bool b,const char* m){if(!b)throw std::runtime_error(m);}
int main(){
  {
    // Regression test for K4 co-moving internal-edge counterexample:
    // K4 (m=6, m2=12), before={0,0,1,1}, after={1,1,1,1}, movers={0,1}.
    // True modularity change is 0 - (-1/6) = +1/6, i.e., m*DeltaQ = 1.0.
    Graph g; g.ensure(4);
    for(int u=0;u<4;u++) for(int v=u+1;v<4;v++) g.add_edge(u,v,1.0);
    std::vector<int> before{0,0,1,1}, after{1,1,1,1}, movers{0,1};
    std::unordered_map<int,double> totals{{0,6.0},{1,6.0}};
    double actual = exact_group_gain(g, before, after, movers, totals);
    double dq = modularity(g, after) - modularity(g, before);
    require(std::fabs(actual - 1.0) < 1e-12, "K4 exact_group_gain regression failed");
    require(std::fabs(exact_group_gain_dense(g, before, after, movers, totals) - 1.0) < 1e-12, "K4 dense regression failed");
    require(std::fabs(dq - 1.0/6.0) < 1e-12, "K4 modularity delta mismatch");
  }
  std::mt19937 rng(20261012);int gain_cases=0,dp_cases=0;
  for(int trial=0;trial<10000;trial++){
    Graph g;int n=2+rng()%11;g.ensure(n);
    for(int u=0;u<n;u++)for(int v=u;v<n;v++)if(rng()%4==0)g.add_edge(u,v,1+rng()%5);
    if(g.m2==0)g.add_edge(0,1,1);
    std::vector<int> before(n),after(n),movers;std::unordered_map<int,double> totals;
    for(int v=0;v<n;v++){before[v]=rng()%4;after[v]=rng()%2?before[v]:rng()%4;totals[before[v]]+=g.k[v];if(before[v]!=after[v])movers.push_back(v);}
    double actual=exact_group_gain(g,before,after,movers,totals),expected=(modularity(g,after)-modularity(g,before))*g.m2/2;
    require(std::fabs(actual-expected)<1e-8,"random joint-gain disagreement");require(exact_group_gain_dense(g,before,after,movers,totals)==actual,"dense-flag gain changed summation");gain_cases++;
  }
  {
    Tree t{{2,2,-1},{4,4,2},{{0,10},{0,20},{1,30}},{{{10,0},{20,1}},{{30,2}}}};
    Ledger l;l.ensure(t);l.begin();
    l.commit(l.quote({},{{{0,10},10,2},{{1,30},10,.5}}));require(l.balance()==2.5,"partial booking");
    l.commit(l.quote({},{{{0,10},10,2},{{1,30},10,.5}}));require(l.balance()==5,"remaining booking");
    require(l.unbooked({0,10},10)==0&&l.unbooked({1,30},10)==0,"repeat booking not suppressed");
    auto q=l.quote({{0,20},{1,30}},{});require(q.delta==5,"shared ancestor retirement wrong");l.commit(q);
    require(l.balance()==4,"ancestor share did not retire");
    q=l.quote({{0,10},{1,30}},{});require(q.delta==0,"booked source charged twice");l.commit(q);
    require(l.quote({{0,10},{1,30}},{}).delta==0,"opened work not marginal zero");
    auto before=l.balance();l.quote({},{{{0,20},20,1}});require(l.balance()==before,"preview mutated ledger");
    require(l.finish(t)==10&&l.balance()==0,"execution conservation");l.begin();
    l.commit(l.quote({},{{{0,10},10,4},{{1,30},10,1}}));
    l.commit(l.quote({}, {},{{0,10}}));require(l.balance()==1,"source cancellation");
    l.schedule({{1,30}},"vanished_parent");require(l.balance()==0,"parent retirement");l.finish(t);l.check();
    require(relative_drift(0,0)==0&&std::isinf(relative_drift(1e-20,0)),"zero reference drift");
  }
  for(int trial=0;trial<1500;trial++){
    int n=1+rng()%9;Tree t;t.parent.resize(n);t.weight.resize(n);t.nodeKey.resize(n);
    std::vector<int> height(n,0);for(int i=0;i<n;i++)t.parent[i]=i?int(rng()%i):-1;
    for(int i=n-1;i>0;i--)height[t.parent[i]]=std::max(height[t.parent[i]],height[i]+1);
    t.ids.resize(1+*std::max_element(height.begin(),height.end()));Ledger l;
    for(int i=0;i<n;i++){t.weight[i]=1+rng()%5;t.nodeKey[i]={height[i],i};t.ids[height[i]][i]=i;l.values[t.nodeKey[i]]={t.weight[i],{},0};}
    IncrementalDP cache;
    for(int round=0;round<4;round++){
      std::map<Key,double> gain;std::vector<AtomicAction> actions;
      double lambda=(rng()%4)*.5;
      for(int i=0;i<n;i++){
        double raw=int(rng()%21)-10; // independently optional actions
        if(raw>0){gain[t.nodeKey[i]]=raw;actions.push_back({raw,{i}});}
      }
      if(round==2)l.opened[t.nodeKey[rng()%n]]=l.weight(t.nodeKey[rng()%n]);
      std::vector<double> weights(n);for(int i=0;i<n;i++)weights[i]=l.marginal(t.nodeKey[i]);
      auto selected=cache.solve(t,gain,l,lambda,true);
      double got=0;for(Key k:selected)got+=(gain.count(k)?gain[k]:0)-lambda*l.marginal(k);
      double best=0;for(int mask=0;mask<(1<<n);mask++){
        bool ok=true;double value=0;for(int i=0;i<n;i++)if(mask&(1<<i)){if(t.parent[i]>=0&&!(mask&(1<<t.parent[i])))ok=false;Key k=t.nodeKey[i];value+=(gain.count(k)?gain[k]:0)-lambda*weights[i];}
        if(ok)best=std::max(best,value);
      }
      auto closure=select_atomic_actions(t.parent,weights,actions,lambda);
      require(std::fabs(got-best)<1e-8&&std::fabs(got-closure.objective)<1e-8,"DP/enumeration/closure disagree");dp_cases++;
    }
  }
  int pair_cases=0;
  for(int trial=0;trial<2000;trial++){
    Graph g;int n=2+rng()%10;g.ensure(n);
    for(int u=0;u<n;u++)for(int v=u+1;v<n;v++)if(rng()%3==0)g.add_edge(u,v,1+rng()%5);
    if(g.m2==0)g.add_edge(0,1,1);
    std::vector<int> before(n);std::unordered_map<int,double> totals;
    for(int v=0;v<n;v++){before[v]=rng()%4;totals[before[v]]+=g.k[v];}
    int u=rng()%n, v=rng()%n; if(u==v) v=(u+1)%n;
    int A1=before[u], B1=rng()%4, A2=before[v], B2=rng()%4;
    if(B1==A1) B1=(A1+1)%4;
    if(B2==A2) B2=(A2+1)%4;
    std::vector<int> after1=before, after2=before, after12=before;
    after1[u]=B1; after2[v]=B2; after12[u]=B1; after12[v]=B2;
    double g1=exact_group_gain(g,before,after1,{u},totals);
    double g2=exact_group_gain(g,before,after2,{v},totals);
    double g12=exact_group_gain(g,before,after12,{u,v},totals);
    double wuv=0.0; for(auto [nb,wt]:g.adj[u]) if((int)nb==v) wuv+=wt;
    int sign=(B1==B2?1:0)+(A1==A2?1:0)-(B1==A2?1:0)-(A1==B2?1:0);
    double inter=(wuv - g.k[u]*g.k[v]/g.m2)*(double)sign;
    require(std::fabs((g12-(g1+g2))-inter)<1e-9,"pairwise interaction identity mismatch");
    pair_cases++;
  }
  std::cout<<"PASS joint_gain="<<gain_cases<<" pair_interaction="<<pair_cases<<" DP/enumeration/mincut="<<dp_cases<<" ledger_lifecycle=12\n";
}
