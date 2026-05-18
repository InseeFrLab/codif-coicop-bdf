#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
DataFrame distance_gcd_batch_cpp(CharacterVector ids,
                                 CharacterVector s1_vec,
                                 CharacterVector s2_vec,
                                 CharacterVector codes) {
  int n = ids.size();
  
  // pré-allouer au max (on filtrera ensuite)
  std::vector<std::string> out_id, out_s1, out_s2, out_code, out_sub;
  std::vector<double> out_dist, out_prop, out_prop_s1;
  out_id.reserve(n); out_s1.reserve(n); out_s2.reserve(n);
  out_code.reserve(n); out_sub.reserve(n);
  out_dist.reserve(n); out_prop.reserve(n); out_prop_s1.reserve(n);
  
  for (int k = 0; k < n; k++) {
    std::string s1 = as<std::string>(s1_vec[k]);
    std::string s2 = as<std::string>(s2_vec[k]);
    int n1 = s1.size(), n2 = s2.size();
    
    // LCS via une seule ligne de DP (O(n2) mémoire au lieu de O(n1*n2))
    std::vector<int> prev(n2 + 1, 0), curr(n2 + 1, 0);
    int maxlen = 0, endpos = 0;
    
    for (int i = 0; i < n1; i++) {
      std::fill(curr.begin(), curr.end(), 0);
      char c1 = toupper(s1[i]);
      for (int j = 0; j < n2; j++) {
        if (c1 == toupper(s2[j])) {
          curr[j + 1] = prev[j] + 1;
          if (curr[j + 1] > maxlen) {
            maxlen = curr[j + 1];
            endpos = i;
          }
        }
      }
      std::swap(prev, curr);
    }
    
    double distance = 1.0 - ((double)maxlen / std::max(n1, n2));
    if (distance > 0.8) continue;
    
    out_id.push_back(as<std::string>(ids[k]));
    out_s1.push_back(s1);
    out_s2.push_back(s2);
    out_code.push_back(as<std::string>(codes[k]));
    out_dist.push_back(distance);
    out_sub.push_back((maxlen > 0) ? s1.substr(endpos - maxlen + 1, maxlen) : "");
    out_prop.push_back((n2 > 0) ? (double)maxlen / n2 : 0.0);
    out_prop_s1.push_back((n1 > 0) ? (double)maxlen / n1 : 0.0);
  }
  
  return DataFrame::create(
    _["id"] = wrap(out_id),
    _["s1"] = wrap(out_s1),
    _["s2"] = wrap(out_s2),
    _["code"] = wrap(out_code),
    _["distance"] = wrap(out_dist),
    _["common_substring"] = wrap(out_sub),
    _["prop_in_s1"] = wrap(out_prop_s1),
    _["prop_in_s2"] = wrap(out_prop),
    _["stringsAsFactors"] = false
  );
}