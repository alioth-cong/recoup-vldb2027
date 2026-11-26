#pragma once
#include <chrono>
#include "corrected_state.hpp"
#include "closure.hpp"
#include "localized_closure.hpp"
#include "compressed_cut.hpp"
namespace dylo_corrected {
struct SelectorStats {std::string solver;double ms=0;size_t candidates=0,eligible=0,pruned=0,active_nodes=0;double objective=0;};
template<class Tree> dylo::LocalizedSelection release_select_dense(const Tree& t,const std::vector<dylo::AtomicAction>& actions,const Ledger& ledger,double lambda,const std::string& mode,bool use_dp,bool prune,IncrementalDP& dp,bool verify,SelectorStats& stats,bool compress_free=true){
  auto started=std::chrono::steady_clock::now();
  dylo::LocalizedSelection out;out.selected_actions.assign(actions.size(),0);out.dirty_nodes.assign(t.parent.size(),0);
  dp.repaired=0;stats.candidates=actions.size();bool single=true;std::map<Key,double> gain;
  std::vector<char> eligible(actions.size(),0),child(t.parent.size(),0);
  for(int p:t.parent)if(p>=0)child[p]=1;
  for(size_t i=0;i<actions.size();i++)if(actions[i].gain>0){eligible[i]=1;single&=actions[i].anchors.size()==1;if(actions[i].anchors.size()==1)gain[t.nodeKey.at(actions[i].anchors[0])]+=actions[i].gain;}
  if(prune&&single){std::set<Key> removed;for(auto [key,g]:gain){int n=t.ids.at(key.first).at(key.second);if(!child[n]&&g<=lambda*ledger.marginal(key))removed.insert(key);}
    for(size_t i=0;i<actions.size();i++)if(eligible[i]&&removed.count(t.nodeKey[actions[i].anchors[0]])){eligible[i]=0;stats.pruned++;}for(Key key:removed)gain.erase(key);}
  stats.eligible=std::count(eligible.begin(),eligible.end(),char(1));
  const bool exact=mode=="dp"||mode=="cut";
  stats.solver=exact?(mode=="cut"||!use_dp||!single?"cut":"dp"):mode;
  if(stats.eligible){
    if(stats.solver=="dp"){
      auto opened=dp.solve(t,gain,ledger,lambda,verify);
      for(size_t i=0;i<actions.size();i++)if(eligible[i]&&opened.count(t.nodeKey[actions[i].anchors[0]]))out.selected_actions[i]=1;
    }else if(stats.solver=="cut"){
      std::vector<dylo::AtomicAction> compact;std::vector<size_t> mapping;std::vector<double> weights(t.parent.size());
      for(size_t n=0;n<weights.size();n++)weights[n]=ledger.marginal(t.nodeKey[n]);
      for(size_t i=0;i<actions.size();i++)if(eligible[i]){compact.push_back(actions[i]);mapping.push_back(i);}
      auto selected=compress_free?compressed_cut(t.parent,weights,compact,lambda):dylo::select_atomic_actions(t.parent,weights,compact,lambda).selected_actions;
      for(size_t j=0;j<mapping.size();j++)out.selected_actions[mapping[j]]=selected[j];
    }else{
      std::map<int,std::vector<size_t>> buckets;
      for(size_t i=0;i<actions.size();i++)if(eligible[i]){
        int bucket=mode=="indep"?int(i):(actions[i].destination>=0?actions[i].destination:actions[i].anchors.at(0));
        buckets[bucket].push_back(i);
      }
      for(auto& [bucket,indices]:buckets){
        double g=0,w=0;std::set<int> path;
        for(size_t i:indices){g+=actions[i].gain;for(int n:actions[i].anchors)for(int c=n;c>=0;c=t.parent[c])path.insert(c);}
        for(int c:path)w+=ledger.marginal(t.nodeKey[c]);
        if(g>lambda*w+1e-12)for(size_t i:indices)out.selected_actions[i]=1;
      }
    }
    for(size_t i=0;i<actions.size();i++)if(out.selected_actions[i]){out.selected_gain+=actions[i].gain;for(int n:actions[i].anchors)for(int c=n;c>=0;c=t.parent[c])out.dirty_nodes[c]=1;}
    for(size_t n=0;n<out.dirty_nodes.size();n++)if(out.dirty_nodes[n])out.marginal_cost+=ledger.marginal(t.nodeKey[n]);
  }
  out.objective=out.selected_gain-lambda*out.marginal_cost;stats.objective=out.objective;
  stats.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();return out;
}
struct ReleaseSelection {
  std::vector<char> selected_actions;
  std::vector<int> dirty_node_ids;
  double selected_gain=0,marginal_cost=0,objective=0;
};
// Materialize only the ancestor union of positive model actions. No vector
// below is sized by the full cost tree. Nonnegative node prices make omitted
// subtrees irrelevant to the maximum-closure objective.
template<class Tree> ReleaseSelection release_select(const Tree& t,const std::vector<dylo::AtomicAction>& actions,const Ledger& ledger,double lambda,const std::string& mode,bool use_dp,bool prune,IncrementalDP& dp,bool verify,SelectorStats& stats,bool compress_free=true){
  auto started=std::chrono::steady_clock::now();stats=SelectorStats{};
  struct CompactTree {
    std::vector<int> parent;
    std::vector<double> weight;
    std::vector<Key> nodeKey;
    std::vector<std::unordered_map<int,int>> ids;
  } compact;
  std::unordered_set<int> touched;
  for(const auto& action:actions)if(action.gain>0)for(int anchor:action.anchors)
    for(int n=anchor;n>=0;n=t.parent.at(n)){
      if(!touched.insert(n).second)break;
      Key key=t.nodeKey.at(n);
      auto found=t.ids.at(key.first).find(key.second);
      if(found==t.ids.at(key.first).end()||found->second!=n)throw std::runtime_error("inactive selector anchor/path");
    }
  std::vector<int> original(touched.begin(),touched.end());
  std::sort(original.begin(),original.end(),[&](int a,int b){return t.nodeKey[a]<t.nodeKey[b];});
  std::unordered_map<int,int> mapping;
  for(size_t i=0;i<original.size();i++)mapping[original[i]]=int(i);
  for(int n:original){
    Key key=t.nodeKey[n];int id=int(compact.nodeKey.size());
    compact.nodeKey.push_back(key);compact.weight.push_back(t.weight[n]);
    compact.parent.push_back(t.parent[n]>=0?mapping.at(t.parent[n]):-1);
    if(compact.ids.size()<=size_t(key.first))compact.ids.resize(key.first+1);
    compact.ids[key.first][key.second]=id;
  }
  std::vector<dylo::AtomicAction> local=actions;
  for(auto& action:local){
    if(action.gain<=0){action.anchors.clear();continue;}
    for(int& anchor:action.anchors)anchor=mapping.at(anchor);
  }
  auto dense=release_select_dense(compact,local,ledger,lambda,mode,use_dp,prune,dp,verify,stats,compress_free);
  ReleaseSelection out;out.selected_actions=std::move(dense.selected_actions);
  out.selected_gain=dense.selected_gain;out.marginal_cost=dense.marginal_cost;out.objective=dense.objective;
  for(size_t i=0;i<original.size();i++)if(dense.dirty_nodes[i])out.dirty_node_ids.push_back(original[i]);
  stats.active_nodes=original.size();
  stats.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
  return out;
}
} // namespace
