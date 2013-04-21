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

    Traits::vertex_descriptor s, t;
    read_dimacs_max_flow(g, capacity, rev, s, t,ifs);

    timer time;
    long flow = push_relabel_max_flow(g, s, t);
    std::cout << "Max flow by push relabel: " << flow << std::endl;
    std::cout << "Elapsed time: " << time.elapsed() << std::endl;
    time.restart();
    long flow2 = boykov_kolmogorov_max_flow(g,s,t);
    std::cout << "Max flow by bk: " << flow2 << std::endl;
    std::cout << "Elapsed time: " << time.elapsed() << std::endl;
    time.restart();
    // long flow3 = edmonds_karp_max_flow(g,s,t);
    // std::cout << "Max flow by edmonds karp: " << flow3 << std::endl;
    // std::cout << "Elapsed time: " << time.elapsed() << std::endl;

    return 0;
}
