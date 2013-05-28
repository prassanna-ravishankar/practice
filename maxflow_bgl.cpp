#include <boost/config.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <boost/graph/push_relabel_max_flow.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/edmonds_karp_max_flow.hpp>
#include <boost/graph/read_dimacs.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/timer.hpp>
#include "bgl_def.hpp"

int
main(int argc, char **argv)
{
    using namespace boost;

    Graph g;
    std::ifstream ifs(argv[1]);
    property_map<Graph, edge_capacity_t>::type
            capacity = get(edge_capacity, g);
    property_map<Graph, edge_reverse_t>::type
            rev = get(edge_reverse, g);
    property_map<Graph, edge_residual_capacity_t>::type
            residual_capacity = get(edge_residual_capacity, g);
    property_map<Graph, vertex_color_t>::type
      color = get(vertex_color, g);
    property_map<Graph, vertex_index_t>::type
      index = get(vertex_index, g);

    Traits::vertex_descriptor s, t;
    read_dimacs_max_flow(g, capacity, rev, s, t,ifs);

    timer time;
    long flow2 = boykov_kolmogorov_max_flow(g,s,t,
                                            capacity_map( capacity )
                                              .reverse_edge_map( rev )
                                              .residual_capacity_map( residual_capacity )
                                              .color_map( color ));

    std::cout << "Max flow by bk: " << flow2 << std::endl;
    time.restart();

    graph_traits<Graph>::vertex_iterator u_iter, u_end;
    graph_traits<Graph>::out_edge_iterator ei, e_end;

    long sum = 0;
    int b = boost::color_traits<int>::black();
    for (boost::tie(u_iter, u_end) = vertices(g); u_iter != u_end; ++u_iter)
    {
        for(boost::tie(ei,e_end) = out_edges(*u_iter,g); ei != e_end; ++ei){
//            if((color[*u_iter] == boost::color_traits<int>::white() || color[*u_iter] == boost::color_traits<int>::gray())
//                    && color[target(*ei, g)] == boost::color_traits<int>::black())
            if(color[*u_iter] == b  && color[target(*ei, g)] != b){
                //std::cout << index[*u_iter] << " to " << index[target(*ei,g)]<< ": " << capacity[*ei] << std::endl;
                sum += capacity[*ei];
            }
        }
    }
    std::cout << sum << std::endl;

    return 0;
}
