#pragma once
#include <chrono>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
namespace dylo_corrected {
struct PhaseProfile {
  std::map<std::string,double> ms;
  std::map<std::string,size_t> count;
  std::ofstream file;
  explicit PhaseProfile(const std::string& path):file(path){file<<"batch\tphase\tms\tcount\n"<<std::setprecision(17);}
  void flush(int batch){for(auto& [phase,time]:ms)file<<batch<<"\t"<<phase<<"\t"<<time<<"\t"<<count[phase]<<"\n";file.flush();ms.clear();count.clear();}
};
struct RoundProfile {
  using Clock=std::chrono::steady_clock;
  PhaseProfile& p;
  const char* phase="round_setup";
  Clock::time_point started=Clock::now();
  explicit RoundProfile(PhaseProfile& profile):p(profile){}
  void mark(const char* next){auto now=Clock::now();p.ms[phase]+=std::chrono::duration<double,std::milli>(now-started).count();p.count[phase]++;phase=next;started=now;}
  ~RoundProfile(){mark("done");}
};
}
