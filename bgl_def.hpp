#include <boost/graph/adjacency_list.hpp>

using namespace boost;

typedef adjacency_list_traits<vecS, vecS, directedS> Traits;
typedef adjacency_list < vecS, vecS, directedS,
             property < vertex_name_t, std::string,
                    property < vertex_index_t, long,
                           property < vertex_color_t, boost::default_color_type,
                              property < vertex_distance_t, long,
                                     property < vertex_predecessor_t, Traits::edge_descriptor > > > > >,

             property < edge_capacity_t, long,
                    property < edge_residual_capacity_t, long,
                           property < edge_reverse_t, Traits::edge_descriptor > > > > Graph;
