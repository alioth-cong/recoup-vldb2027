#pragma once
#include "closure.hpp"
namespace dylo_corrected {
// Zero-price reports impose no penalty; bypass them but retain every priced
// ancestor dependency. Action nodes and their original order are unchanged.
inline std::vector<char> compressed_cut(const std::vector<int>& parent,const std::vector<double>& weight,
                                       const std::vector<dylo::AtomicAction>& actions,double lambda){
  const int n=parent.size();std::vector<int> nearest(n,-2),mapping(n,-1),priced_parent;
  std::vector<double> priced_weight;
  for(int start=0;start<n;start++)if(nearest[start]==-2){
    std::vector<int> path;int c=start;
    while(c>=0&&nearest[c]==-2){path.push_back(c);c=parent[c];}
    int next=c>=0?nearest[c]:-1;
    for(auto it=path.rbegin();it!=path.rend();++it){if(weight[*it]>0)next=*it;nearest[*it]=next;}
  }
  for(int c=0;c<n;c++)if(weight[c]>0){mapping[c]=priced_weight.size();priced_weight.push_back(weight[c]);}
  for(int c=0;c<n;c++)if(weight[c]>0){int p=parent[c]>=0?nearest[parent[c]]:-1;priced_parent.push_back(p>=0?mapping[p]:-1);}
  auto reduced=actions;
  for(auto& action:reduced){
    std::vector<int> anchors;
    for(int c:action.anchors)if(nearest[c]>=0)anchors.push_back(mapping[nearest[c]]);
    std::sort(anchors.begin(),anchors.end());anchors.erase(std::unique(anchors.begin(),anchors.end()),anchors.end());
    action.anchors=std::move(anchors);
  }
  return dylo::select_atomic_actions(priced_parent,priced_weight,reduced,lambda).selected_actions;
}
}
