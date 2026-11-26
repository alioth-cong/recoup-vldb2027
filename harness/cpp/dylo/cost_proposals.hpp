#pragma once
#include "corrected_state.hpp"

namespace dylo_corrected {
// Frozen-round eager quotes. Cache each ancestor path and its ledger prices;
// merge in Key order, exactly the order used by Ledger::quote's std::set.
// Call next_round after ledger/tree mutations and before the next quote.
template<class Tree> class EagerQuoteCache {
  struct Price {Key key;double execution,retirement;};
  const Tree& tree;
  const Ledger& ledger;
  struct CachedPath{size_t epoch=0;std::vector<Price> prices;};
  std::unordered_map<int,CachedPath> paths;
  size_t epoch=1;
  const std::vector<Price>& path(int node){
    auto& cached=paths[node];if(cached.epoch==epoch)return cached.prices;
    auto& result=cached.prices;result.clear();
    for(int n=node;n>=0;n=tree.parent[n]){
      Key k=tree.nodeKey[n];bool opened=ledger.opened.count(k);
      result.push_back({k,opened?0:ledger.weight(k),opened?0:ledger.pending_at(k)});
    }
    std::sort(result.begin(),result.end(),[](const Price& a,const Price& b){return a.key<b.key;});
    cached.epoch=epoch;return result;
  }
 public:
  EagerQuoteCache(const Tree& t,const Ledger& l):tree(t),ledger(l){}
  void next_round(){++epoch;}
  double delta(int source,int destination,bool source_disappears){
    const auto& a=path(source);const auto& b=path(destination);
    size_t i=0,j=0;double execution=0,retirement=0;
    bool gone=source_disappears&&source>=0;
    Key removed=gone?tree.nodeKey[source]:Key{-1,-1};
    while(i<a.size()||j<b.size()){
      Price price;
      if(j==b.size()||(i<a.size()&&a[i].key<b[j].key))price=a[i++];
      else if(i==a.size()||b[j].key<a[i].key)price=b[j++];
      else{price=a[i++];j++;}
      if(!gone||price.key!=removed){execution+=price.execution;retirement+=price.retirement;}
    }
    return execution-retirement-(gone?ledger.pending_at(removed):0);
  }
};
// A frozen-round singleton quote. Source ownership is not divided among
// hypothetical moves: this prices the actual alternative being compared.
template<class Tree>
double singleton_delta(const Tree& t,const Ledger& ledger,int source,int destination,
                       bool source_disappears,bool defer,bool source_charge) {
  std::set<Key> paths,gone;
  std::vector<Share> shares;
  auto add_path=[&](int n){for(int x=n;x>=0;x=t.parent[x])paths.insert(t.nodeKey[x]);};
  if(destination>=0)add_path(destination);
  if(source>=0){
    int owner=t.nodeKey[source].second;
    if(!defer)add_path(source);
    else if(source_charge)for(int x=source;x>=0;x=t.parent[x]){
      Key key=t.nodeKey[x];shares.push_back({key,owner,ledger.unbooked(key,owner)});
    }
    if(source_disappears)gone.insert(t.nodeKey[source]);
  }
  return ledger.quote(paths,shares,gone).delta;
}

// Keep a raw-gain fallback for shared-cost groups whose singleton net gains
// are all nonpositive. This remains a one-proposal-per-vertex heuristic.
struct DestinationChoice {
  int raw=-1,priced=-1;
  double raw_gain=0,priced_gain=0,net=0;
  void consider(int destination,double gain,double cost,double lambda){
    if(gain<=1e-12)return;
    if(gain>raw_gain+1e-12||(std::fabs(gain-raw_gain)<=1e-12&&(raw<0||destination<raw))){raw=destination;raw_gain=gain;}
    double value=gain-lambda*cost;
    if(value<=1e-12)return;
    if(value>net+1e-12||(std::fabs(value-net)<=1e-12&&
       (gain>priced_gain+1e-12||(std::fabs(gain-priced_gain)<=1e-12&&(priced<0||destination<priced))))){
      priced=destination;priced_gain=gain;net=value;
    }
  }
  int destination()const{return priced>=0?priced:raw;}
  double gain()const{return priced>=0?priced_gain:raw_gain;}
};
}
