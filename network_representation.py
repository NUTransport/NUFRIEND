"""Simplify, correct, and consolidate network topology.
Taken from OSMNX simplification module. Edited by AHT8237.
"""

from util import *
# MODULES
from helper import RR_line, vals_str

sns.set_palette('colorblind', n_colors=4)

'''
LOAD GRAPHS
'''


def load_simplified_consolidated_graph(rr: str, radius=10000, intertypes=None) -> nx.Graph:

    gpkl_path = os.path.join(NX_DIR, rr + '_geo_graph_simplified.pkl')
    if os.path.exists(gpkl_path):
        G = nx.read_gpickle(gpkl_path)

    else:
        G = load_graph(rr)

        G = subgraph_from_interchange_nodes_geo(G, intertypes=intertypes)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            G = consolidate_graph(G, radius=radius)

        G = augment_stations_splc(G)

        G = augment_node_city_name(G)

        # nodes_gdf, edges_gdf = gdfs_from_graph(G, smooth_geometry=True)
        #
        # G = undirected_graph_from_gdfs(gdf_nodes=nodes_gdf, gdf_edges=edges_gdf, graph_attrs=G.graph)

        nx.write_gpickle(G, gpkl_path)

    return G


def undirected_graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs=None) -> nx.Graph:
    """
    Convert nodes and links geodataframes into a network representation using netwokrx
    :param rr: [str] railroad of interest; will remove all other railroads
    :return: [nx.MultiGraph] undirected graph with node id's used to reference nodes and edges;
                        features from geodataframes will be stored in the graph node and edge attributes
    """

    if graph_attrs is None:
        graph_attrs = {"crs": gdf_edges.crs}
    Gun = nx.Graph(**graph_attrs)

    node_cols = gdf_nodes.columns
    edge_cols = gdf_edges.columns
    # for each link entry in the link geodataframe; add new edges and nodes with information
    for u, v in gdf_edges.index:
        edge_data = gdf_edges.loc[(u, v)]  # gdf row of edge e's features
        u_data = gdf_nodes.loc[u]  # gdf row of node u's features
        v_data = gdf_nodes.loc[v]  # gdf row of node v's features

        # check if nodes have not already been added
        if u not in Gun:
            Gun.add_nodes_from([(u, {a: u_data[a] for a in node_cols})])
        if v not in Gun:
            Gun.add_nodes_from([(v, {a: v_data[a] for a in node_cols})])

        # features to add as edge attributes
        Gun.add_edges_from([(u, v, {a: edge_data[a] for a in edge_cols})])

    return Gun


def digraph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs=None) -> nx.DiGraph:
    """
    Convert node and edge GeoDataFrames to a MultiDiGraph.

    This function is the inverse of `graph_to_gdfs` and is designed to work in
    conjunction with it. However, you can convert arbitrary node and edge
    GeoDataFrames as long as gdf_nodes is uniquely indexed by `osmid` and
    gdf_edges is uniquely multi-indexed by `u`, `v`, `key` (following normal
    MultiDiGraph structure). This allows you to load any node/edge shapefiles
    or GeoPackage layers as GeoDataFrames then convert them to a MultiDiGraph
    for graph analysis.

    Parameters
    ----------
    gdf_nodes : geopandas.GeoDataFrame
        GeoDataFrame of graph nodes uniquely indexed by osmid
    gdf_edges : geopandas.GeoDataFrame
        GeoDataFrame of graph edges uniquely multi-indexed by u, v, key
    graph_attrs : dict
        the new G.graph attribute dict. if None, use crs from gdf_edges as the
        only graph-level attribute (gdf_edges must have crs attribute set)

    Returns
    -------
    G : networkx.MultiDiGraph
    """
    if gdf_nodes is not None and gdf_edges is not None:
        if graph_attrs is None:
            graph_attrs = {"crs": gdf_edges.crs}
        G = nx.DiGraph(**graph_attrs)

        # add edges and their attributes to graph, but filter out null attribute
        # values so that edges only get attributes with non-null values
        attr_names = gdf_edges.columns.to_list()
        for (u, v), attr_vals in zip(gdf_edges.index, gdf_edges.values):
            data_all = zip(attr_names, attr_vals)
            data = {name: val for name, val in data_all if isinstance(val, list) or pd.notnull(val)}
            G.add_edge(u, v, **data)

        # add nodes' attributes to graph
        for col in gdf_nodes.columns:
            nx.set_node_attributes(G, name=col, values=gdf_nodes[col].dropna())

        return G

    return None


def load_graph(rr: str) -> nx.Graph:

    gpkl_path = os.path.join(NX_DIR, rr + '_geo_graph.pkl')
    if os.path.exists(gpkl_path):
        G = nx.read_gpickle(gpkl_path)
    else:
        # load geodataframes from geojson files for specific railroad
        gdf_n, gdf_l = load_line_node_gdf(rr)

        # links
        gdf_l = gdf_l[['linkid', 'fromnode', 'tonode', 'density', 'miles', 'km', 'geometry']]  # extract attributes
        gdf_l.rename(columns={'fromnode': 'u', 'tonode': 'v'}, inplace=True)  # naming convention for ox
        gdf_l['length'] = gdf_l['km']  # add length attribute, in km
        gdf_l['osmid'] = gdf_l['linkid']  # add osmid attribute, same as linkid
        gdf_l = gdf_l.groupby(by=['u', 'v'], as_index=True).first()  # index by edge label multi-index
        # nodes
        gdf_n = gdf_n[['nodeid', 'name', 'splc', 'franodeid',
                       'inttype', 'lon', 'lat', 'geometry']]  # extract attributes
        gdf_n.rename(columns={'nodeid': 'osmid', 'lon': 'x', 'lat': 'y'}, inplace=True)  # naming convention for ox
        gdf_n.index = gdf_n['osmid']  # reindex by osmid
        # graph attributes
        graph_attrs = {'railroad': rr, 'crs': gdf_l.crs}
        # load these GeoDataFrames into an OSMNX MultiDiGraph
        G = undirected_graph_from_gdfs(gdf_n, gdf_l, graph_attrs=graph_attrs)
        # write to pkl file
        nx.write_gpickle(G, gpkl_path)

    return G



'''
GRAPH PROCESSING: 
    - Simplification
    - Node intersection consolidation
    - Coordinate projection
    - Subgraph induction
    - Feature augmentation
'''


def subgraph_from_interchange_nodes_geo(G: nx.Graph, intertypes=None, remove_nodes=None,
                                        max_connected=True, simplify_geometry_tolerance=0.01) -> nx.Graph:
    # G is directed
    # return subgraph of G that includes only the nodes with field 'inttype' in priority_type list

    G_p = G.copy()  # get copy to not change G

    if intertypes is None:
        intertypes = {'T', 'P'}

    if remove_nodes is None:
        remove_nodes = [n for n in G.nodes() if G.nodes[n]['inttype'] not in intertypes]

    for u, v, d in G_p.edges(data=True):
        G_p.edges[u, v]['original_linkids'] = [d['linkid']]

    for n in remove_nodes:
        neighbors = set(G_p[n])  # get neighbors of node n

        for u in neighbors:
            # for each neighboring node add an edge between it and all other neighboring nodes of n
            for v in neighbors.difference({u}):
                # ensures u != v
                # if there is not already an existing edge between u and v
                # G_p.has_edge(u, v) == G_p.has_edge(v, u) and G_p.has_edge(u, u) == False for undirected Graphs
                if not G_p.has_edge(u, v):
                    # join edges (u, n) and (n, v) with summed distance to route through n
                    miles = G_p.edges[u, n]['miles'] + G_p.edges[n, v]['miles']
                    km = G_p.edges[u, n]['km'] + G_p.edges[n, v]['km']
                    # join edge geometries of (u, n) and (n, v)
                    geom = linemerge([G_p.edges[u, n]['geometry'], G_p.edges[n, v]['geometry']])
                    if type(geom) is MultiLineString:
                        # if type MultiLineString, convert to single LineString
                        geom = LineString([point for line in list(geom.geoms) for point in line.coords])
                    geom = geom.simplify(simplify_geometry_tolerance)
                    # create new linkid to indicate it is a super link
                    linkid = str(G_p.edges[u, n]['linkid'])
                    linkid = linkid if str(G_p.edges[u, n]['linkid'])[0] == 'S' else 'S' + linkid
                    # combine list of linkids that make up new link- for routing
                    un_linkids = G_p.edges[u, n]['original_linkids']
                    nv_linkids = G_p.edges[n, v]['original_linkids']
                    new_linkids = un_linkids + nv_linkids
                    # add new edge (u, v) to graph with updated (joined) attributes of each previous edge
                    G_p.add_edge(u, v, miles=miles, length=km, km=km, geometry=geom, linkid=linkid, osmid=linkid,
                                 original_linkids=new_linkids)

        # remove n from G; removes all attached edges
        G_p.remove_node(n)

    # extract largest connected component from graph
    if max_connected:
        G_p = G_p.subgraph(max(nx.connected_components(G_p), key=len)).copy()

    return G_p


def consolidate_graph(G: nx.Graph, radius=None, crs='epsg:5070'):
    # Consolidate nodes into one super node by buffer radius in meters (?)

    # def _merge_nodes_geometric(G, tolerance):
    """
    Geometrically merge nodes within some distance of each other.

    If chunk=True, it sorts the nodes GeoSeries by geometry x and y values (to
    make unary_union faster), then buffers by tolerance. Next it divides the
    nodes GeoSeries into n-sized chunks, where n = the square root of the
    number of nodes. Then it runs unary_union on each chunk, and then runs
    unary_union on the resulting unary unions. This is much faster on large
    graphs (n>100000) because of how unary_union's runtime scales with vertex
    count. But chunk=False is usually faster on small and medium sized graphs.
    This hacky method will hopefully be made obsolete when shapely becomes
    vectorized by incorporating the pygeos codebase.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        a projected graph
    tolerance : float
        nodes are buffered to this distance (in graph's geometry's units) and
        subsequent overlaps are dissolved into a single polygon
    chunk : bool
        if True, divide nodes into geometrically sorted chunks to improve the
        speed of unary_union operation by running it on each chunk and then
        running it on the results of those runs

    Returns
    -------
    merged_nodes : GeoSeries
        the merged overlapping polygons of the buffered nodes
    """

    if radius is None:
        radius = 10000

    original_crs = G.graph['crs']
    G = project_graph(G, to_crs=crs)

    gs_nodes = gdfs_from_graph(G, crs=crs, edges=False)['geometry']

    # buffer nodes GeoSeries then get unary union to merge overlaps
    merged_nodes = gs_nodes.buffer(radius).unary_union

    # if only a single node results, make it iterable to convert to GeoSeries
    if isinstance(merged_nodes, Polygon):
        merged_nodes = [merged_nodes]

    merged_nodes = gpd.GeoSeries(list(merged_nodes), crs=G.graph["crs"])

    # STEP 1
    # buffer nodes to passed-in distance and merge overlaps. turn merged nodes
    # into gdf and get centroids of each cluster as x, y
    # AHT8237 -> edit code line: added <chunk=False> to <_merge_nodes_geometric> method call
    node_clusters = gpd.GeoDataFrame(geometry=merged_nodes)
    centroids = node_clusters.centroid
    node_clusters["x"] = centroids.x
    node_clusters["y"] = centroids.y

    # STEP 2
    # attach each node to its cluster of merged nodes. first get the original
    # graph's node points then spatial join to give each node the label of
    # cluster it's within
    node_points = gdfs_from_graph(G, crs=crs, edges=False)[["geometry"]]
    gdf = gpd.sjoin(node_points, node_clusters, how="left", predicate="within")
    gdf = gdf.rename(columns={"index_right": "cluster"})
    gdf['nodeid'] = gdf.index
    gdf['centroid'] = gdf['nodeid'].apply(lambda u: Point(gdf.loc[u, 'x'], gdf.loc[u, 'y']))

    # STEP 3
    # if a cluster contains multiple components (i.e., it's not connected)
    # move each component to its own cluster (otherwise you will connect
    # nodes together that are not truly connected, e.g., nearby deadends or
    # surface streets with bridge).
    groups = gdf.groupby("cluster")
    for cluster_label, nodes_subset in groups:
        if len(nodes_subset) > 1:
            # identify all the (connected) components in cluster
            ccs = list(nx.connected_components(G.subgraph(nodes_subset.index)))
            if len(ccs) > 1:
                # if there are multiple components in this cluster
                suffix = 0
                for cc in ccs:
                    cc = list(cc)
                    # set subcluster xy to the centroid of just these nodes
                    subcluster_centroid = node_points.loc[cc].unary_union.centroid
                    gdf.loc[cc, "x"] = subcluster_centroid.x
                    gdf.loc[cc, "y"] = subcluster_centroid.y
                    # move to subcluster by appending suffix to cluster label
                    gdf.loc[cc, "cluster"] = f"{cluster_label}-{suffix}"
                    suffix += 1

    # STEP 4
    # create new empty graph and copy over misc graph data
    H = nx.Graph()
    H.graph = G.graph

    # STEP 5
    supernode_dict = dict()
    # create a new node for each cluster of merged nodes
    # regroup now that we potentially have new cluster labels from step 3
    groups = gdf.groupby("cluster")
    for cluster_label, nodes_subset in groups:
        osmids = nodes_subset.index.to_list()
        center_node = max(osmids, key=G.degree)
        super_node = str(center_node) if str(center_node)[0] == 'S' else 'S' + str(center_node)
        for n in osmids:
            supernode_dict[n] = super_node

        # AHT8237 code block edit:
        # make OSMIDS list, not str, and keep track of SPLC codes and names as well
        # make the inttype of the supernode the dominant type of a facility in the supernode grouping
        inttypes = {G.nodes[o]['inttype'] for o in osmids}
        inttype = 'O'
        if 'T' in inttypes:
            inttype = 'T'
        elif 'P' in inttypes:
            inttype = 'P'
        attr_dict = {'osmid': super_node, 'nodeid': super_node, 'original_nodeids': osmids,
                     'splc': [G.nodes[o]['splc'] for o in osmids],
                     'franodeid': [G.nodes[o]['franodeid'] for o in osmids],
                     'inttype': inttype,
                     'name': [G.nodes[o]['name'] for o in osmids]}
        x = G.nodes[center_node]['x']
        y = G.nodes[center_node]['y']
        H.add_node(super_node, **attr_dict, x=x, y=y, lon=x, lat=y, geometry=Point(x, y))

    # STEP 6
    # create new edge from cluster to cluster for each edge in original graph
    # gdf_edges = gdfs_from_graph(G, nodes=False)
    for u, v, data in G.edges(data=True):
        u2 = supernode_dict[u]
        v2 = supernode_dict[v]
        # only create the edge if we're not connecting the cluster to itself
        if u2 != v2:
            data["u_original"] = u
            data["v_original"] = v
            H.add_edge(u2, v2, **data)

    # STEP 7
    # for every group of merged nodes with more than 1 node in it, extend the
    # edge geometries to reach the new node point
    for cluster_label, nodes_subset in groups:
        osmids = nodes_subset.index.to_list()
        # but only if there were multiple nodes merged together,
        # otherwise it's the same old edge as in original graph
        if len(osmids) > 1:
            # get coords of merged nodes point centroid to prepend or
            # append to the old edge geom's coords
            super_node = supernode_dict[osmids[0]]
            x = H.nodes[super_node]["x"]
            y = H.nodes[super_node]["y"]
            xy = [(x, y)]

            # for each edge incident to this new merged node, update its
            # geometry to extend to/from the new node's point coords
            for u, v, d in H.edges([super_node], data=True):
                old_coords = list(d["geometry"].coords)
                wrap_xy = xy + old_coords + xy
                if LineString(wrap_xy[:2]).length < LineString(wrap_xy[-2:]).length:
                    new_coords = wrap_xy[:-1]
                else:
                    new_coords = wrap_xy[1:]
                # add in new geometry for edge
                new_geom = LineString(new_coords)
                H.edges[u, v]["geometry"] = new_geom
                # update distance in miles with new segment distance
                H.edges[u, v]["miles"] = H.edges[u, v]["miles"]
                H.edges[u, v]["km"] = H.edges[u, v]["km"]
                H.edges[u, v]["length"] = H.edges[u, v]["length"]

    H.graph['radius'] = radius

    return project_graph(H, to_crs=original_crs)


def smooth_base_graph_geometry(rr: str, smooth_geometry_tolerance=0.05):

    gpkl_original_path = os.path.join(NX_DIR, rr + '_geo_graph_simplified_original.pkl')
    if os.path.exists(gpkl_original_path):
        G = nx.read_gpickle(gpkl_original_path)

        for u, v in G.edges:
            G.edges[u, v]['geometry'] = G.edges[u, v]['geometry'].simplify(smooth_geometry_tolerance)

        nx.write_gpickle(G, os.path.join(NX_DIR, rr + '_geo_graph_simplified.pkl'))

    return G


def project_graph(G, to_crs=None, smooth_geometry=False, smooth_geometry_tolerance=0.05):
    """
    Project graph from its current CRS to another.

    If to_crs is None, project the graph to the UTM CRS for the UTM zone in
    which the graph's centroid lies. Otherwise, project the graph to the CRS
    defined by to_crs.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        the graph to be projected
    to_crs : string or pyproj.CRS
        if None, project graph to UTM zone in which graph centroid lies,
        otherwise project graph to this CRS

    Returns
    -------
    G_proj : networkx.MultiDiGraph
        the projected graph
    """

    # if isinstance(G.graph['crs'], str):
    #     if G.graph['crs'] == to_crs:
    #         return G
    # elif G.graph['crs'].name.replace(' ', '') == to_crs:
    #     return G
    #
    # # STEP 1: PROJECT THE NODES
    # gdf_nodes = gdfs_from_graph(G, edges=False)
    #
    # # create new lat/lng columns to preserve lat/lng for later reference if
    # # cols do not already exist (ie, don't overwrite in later re-projections)
    # if "lon" not in gdf_nodes.columns or "lat" not in gdf_nodes.columns:
    #     gdf_nodes["lon"] = gdf_nodes["x"]
    #     gdf_nodes["lat"] = gdf_nodes["y"]
    #
    # # project the nodes GeoDataFrame and extract the projected x/y values
    # gdf_nodes_proj = ox.project_gdf(gdf_nodes, to_crs=to_crs)
    # gdf_nodes_proj["x"] = gdf_nodes_proj["geometry"].x
    # gdf_nodes_proj["y"] = gdf_nodes_proj["geometry"].y
    # # gdf_nodes_proj = gdf_nodes_proj.drop(columns=["geometry"])
    #
    # # STEP 2: PROJECT THE EDGES
    # # if "simplified" in G.graph and G.graph["simplified"]:
    # #     # if graph has previously been simplified, project the edge geometries
    # gdf_edges = gdfs_from_graph(G, nodes=False)
    # gdf_edges_proj = ox.project_gdf(gdf_edges, to_crs=to_crs)

    gdf_nodes_proj, gdf_edges_proj = gdfs_from_graph(G, crs=to_crs, smooth_geometry=smooth_geometry,
                                                     smooth_geometry_tolerance=smooth_geometry_tolerance)
    # STEP 3: REBUILD GRAPH
    # turn projected node/edge gdfs into a graph and update its CRS attribute
    if G.is_directed():
        G_proj = digraph_from_gdfs(gdf_nodes_proj, gdf_edges_proj, G.graph)
    else:
        G_proj = undirected_graph_from_gdfs(gdf_nodes_proj, gdf_edges_proj, G.graph)
    G_proj.graph["crs"] = gdf_nodes_proj.crs

    return G_proj


def project_nodes_gdf(nodes_gdf: gpd.GeoDataFrame, to_crs: str = 'WGS84'):

    if nodes_gdf.crs.name.replace(' ', '') == to_crs:
        return nodes_gdf

    # PROJECT THE NODES
    # create new lat/lng columns to preserve lat/lng for later reference if
    # cols do not already exist (ie, don't overwrite in later re-projections)
    # gdf_nodes_proj = gdf_nodes_proj.drop(columns=["geometry"])

    if "lon" not in nodes_gdf.columns or "lat" not in nodes_gdf.columns:
        nodes_gdf["lon"] = nodes_gdf["x"]
        nodes_gdf["lat"] = nodes_gdf["y"]

    # project the nodes GeoDataFrame and extract the projected x/y values
    nodes_gdf_proj = ox.project_gdf(nodes_gdf, to_crs=to_crs)
    nodes_gdf_proj["x"] = nodes_gdf_proj["geometry"].x
    nodes_gdf_proj["y"] = nodes_gdf_proj["geometry"].y

    return nodes_gdf_proj


def project_edges_gdf(edges_gdf: gpd.GeoDataFrame, to_crs: str = 'WGS84'):

    # PROJECT THE EDGES
    # if "simplified" in G.graph and G.graph["simplified"]:
    #     # if graph has previously been simplified, project the edge geometries
    # edges_gdf = gdfs_from_graph(G, nodes=False)

    if edges_gdf.crs.name.replace(' ', '') == to_crs:
        return edges_gdf

    edges_gdf_proj = ox.project_gdf(edges_gdf, to_crs=to_crs)

    return edges_gdf_proj


def project_gdfs(nodes_gdf: gpd.GeoDataFrame=None, edges_gdf: gpd.GeoDataFrame=None, to_crs=None):
    """
    Project graph from its current CRS to another.

    If to_crs is None, project the graph to the UTM CRS for the UTM zone in
    which the graph's centroid lies. Otherwise, project the graph to the CRS
    defined by to_crs.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        the graph to be projected
    to_crs : string or pyproj.CRS
        if None, project graph to UTM zone in which graph centroid lies,
        otherwise project graph to this CRS

    Returns
    -------
    G_proj : networkx.MultiDiGraph
        the projected graph
    """

    if nodes_gdf is not None and edges_gdf is None:
        return project_nodes_gdf(nodes_gdf=nodes_gdf, to_crs=to_crs)
    elif nodes_gdf is None and edges_gdf is not None:
        return project_edges_gdf(edges_gdf=edges_gdf, to_crs=to_crs)
    else:
        return project_nodes_gdf(nodes_gdf=nodes_gdf, to_crs=to_crs), \
               project_edges_gdf(edges_gdf=edges_gdf, to_crs=to_crs)


def remove_from_graph(G: nx.Graph, nodes_to_remove=None, edges_to_remove=None, connected_only=False) -> nx.Graph:
    G = G.copy()

    if nodes_to_remove is None:
        nodes_to_remove = []
    if edges_to_remove is None:
        edges_to_remove = []

    G.remove_edges_from(edges_to_remove)
    G.remove_nodes_from(nodes_to_remove)

    if connected_only:
        # extract largest connected component from graph
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    return G


def augment_stations_splc(G: nx.Graph):
    """
    Augment G's node attributes for SPLC's with additional SPLC data from <filename> file
    by assigning to geographically closest nodes on G
    """

    G = G.copy()

    df_splc = load_splc_station_master()

    gdf_splc = gpd.GeoDataFrame(df_splc, geometry=gpd.points_from_xy(df_splc['lon'], df_splc['lat'],
                                                                     crs=G.graph['crs']))

    gdf_nodes, gdf_edges = gdfs_from_graph(G)

    # project to a 2-D projection prior this
    crs = 'epsg:5070'
    gdf_splc.to_crs(crs=crs, inplace=True)
    gdf_nodes.to_crs(crs=crs, inplace=True)
    gdf_join = gpd.sjoin_nearest(gdf_splc.to_crs(crs=crs), gdf_nodes.to_crs(crs=crs),
                                 how='left', max_distance=5e5, distance_col='distance')
    gdf_join.to_crs(crs=G.graph['crs'], inplace=True)
    gdf_join.dropna(inplace=True)

    gdf_join['SPLC'] = gdf_join['SPLC'].astype(int)
    gdf_join['SPLC'] = gdf_join['SPLC'].apply(lambda x: str(x) if len(str(x)) > 5 else '0' + str(x))
    gdf_join.index = gdf_join['SPLC']
    gdf_join.drop_duplicates(subset='SPLC', inplace=True)

    for splc in gdf_join.index:
        node = gdf_join.loc[splc, 'nodeid']
        splc_set = set(G.nodes[node]['splc'])
        G.nodes[node]['splc'] = list(splc_set.union(set([splc])))

    return G


def augment_node_city_name(G: nx.Graph):
    """
    Augment G's node attributes for SPLC's with additional SPLC data from <filename> file
    by assigning to geographically closest nodes on G
    """

    G = G.copy()

    df_city = load_us_cities()

    gdf_city = gpd.GeoDataFrame(df_city, geometry=gpd.points_from_xy(df_city['lng'], df_city['lat'],
                                                                     crs=G.graph['crs']))

    gdf_nodes, gdf_edges = gdfs_from_graph(G)

    # project to a 2-D projection prior this
    crs = 'epsg:5070'
    # gdf_city.to_crs(crs=crs, inplace=True)
    # gdf_nodes.to_crs(crs=crs, inplace=True)
    gdf_join = gpd.sjoin_nearest(gdf_nodes.to_crs(crs=crs), gdf_city.to_crs(crs=crs),
                                 how='left', max_distance=5e5, distance_col='distance')
    gdf_join.to_crs(crs=G.graph['crs'], inplace=True)
    # gdf_join.dropna(inplace=True)

    # gdf_join.index = gdf_join['SPLC']
    # gdf_join.drop_duplicates(subset='SPLC', inplace=True)

    for node in gdf_join.index:
        G.nodes[node]['city'] = gdf_join.loc[node, 'city']
        G.nodes[node]['state'] = gdf_join.loc[node, 'state_id']

    return G


'''
LOAD GEODATAFRAMES
'''


def agg_rrs_gis(new_rr_name: str, rrs: list):
    # USA1 = ['BNSF', 'NS', 'KCS']

    # load GeoDataFrames for railroad network lines and nodes
    if 'CP' in rrs:
        rrs.extend(['CPUS', 'CPRS', 'SOO'])
    if 'CN' in rrs:
        rrs.extend(['CNUS', 'CNRS', 'GTC'])

    # load geodataframe from geojson file for just lines
    gdf_l = gpd.read_file(os.path.join(NX_DIR, 'Routable_Rail_Lines_Only_Lines.geojson'))  # lines
    # RR_col = ['rrowner1']
    # RR_col = ['rrowner1', 'rrowner2', 'rrowner3']
    RR_col = ['rrowner1', 'rrowner2', 'rrowner3', 'tr1', 'tr2', 'tr3', 'tr4', 'tr5', 'tr6', 'tr7', 'tr8',
              'tr9', 'hr1', 'hr2', 'hr3', 'hr4', 'hr5']
    gdf_l[RR_col] = gdf_l[RR_col].replace(rrs, new_rr_name)

    # filter out all lines not relevant to railroad of interest's network
    gdf_l = gdf_l[RR_line(gdf_l, new_rr_name)].copy()
    gdf_l.index = gdf_l['linkid'].copy()  # reindex gdf with linkid as indices

    # extract unique nodeset of rr network
    nodes_rr = set(gdf_l['fromnode']).union(set(gdf_l['tonode']))

    # load geodataframes from geojson files
    gdf_n = gpd.read_file(os.path.join(NX_DIR, 'Routable_Rail_Lines_Only_Nodes.geojson'))  # nodes

    # keep only nodes in railroad's network
    gdf_n = gdf_n[gdf_n['nodeid'].isin(nodes_rr)].copy()
    gdf_n.index = gdf_n['nodeid'].copy()  # reindex gdf with nodeid as indices

    # graph construction
    # links
    gdf_l = gdf_l[['linkid', 'fromnode', 'tonode', 'density', 'miles', 'km', 'geometry']]  # extract attributes
    gdf_l.rename(columns={'fromnode': 'u', 'tonode': 'v'}, inplace=True)  # naming convention for ox
    gdf_l['length'] = gdf_l['km']  # add length attribute, in km
    gdf_l['osmid'] = gdf_l['linkid']  # add osmid attribute, same as linkid
    gdf_l = gdf_l.groupby(by=['u', 'v'], as_index=True).first()  # index by edge label multi-index
    # nodes
    gdf_n = gdf_n[['nodeid', 'name', 'splc', 'franodeid',
                   'inttype', 'lon', 'lat', 'geometry']]  # extract attributes
    gdf_n.rename(columns={'nodeid': 'osmid', 'lon': 'x', 'lat': 'y'}, inplace=True)  # naming convention for ox
    gdf_n.index = gdf_n['osmid']  # reindex by osmid
    # graph attributes
    graph_attrs = {'railroad': new_rr_name, 'crs': gdf_l.crs}
    # load these GeoDataFrames into an OSMNX MultiDiGraph
    G = undirected_graph_from_gdfs(gdf_n, gdf_l, graph_attrs=graph_attrs)
    # write to pkl file
    gpkl_path = os.path.join(NX_DIR, new_rr_name + '_geo_graph.pkl')
    nx.write_gpickle(G, gpkl_path)


def load_line_node_gdf(rr: str):
    # load GeoDataFrames for railroad network lines and nodes

    # load geodataframe from geojson file for just lines
    gdf_l = gpd.read_file(os.path.join(NX_DIR, 'Routable_Rail_Lines_Only_Lines.geojson'))  # lines

    if rr == 'CN':
        rr = 'CNUS'
    elif rr == 'CP':
        rr = 'CPRS'
    # filter out all lines not relevant to railroad of interest's network
    gdf_l = gdf_l[RR_line(gdf_l, rr)].copy()
    gdf_l.index = gdf_l['linkid'].copy()  # reindex gdf with linkid as indices

    # extract unique nodeset of rr network
    nodes_rr = set(gdf_l['fromnode']).union(set(gdf_l['tonode']))

    # load geodataframes from geojson files
    gdf_n = gpd.read_file(os.path.join(NX_DIR, 'Routable_Rail_Lines_Only_Nodes.geojson'))  # nodes

    # keep only nodes in railroad's network
    gdf_n = gdf_n[gdf_n['nodeid'].isin(nodes_rr)].copy()
    gdf_n.index = gdf_n['nodeid'].copy()  # reindex gdf with nodeid as indices

    return gdf_n, gdf_l


def gdfs_from_graph(G: nx.Graph, nodes=True, edges=True, crs: str = 'WGS84', smooth_geometry=False,
                    smooth_geometry_tolerance=0.05):
    # smooth_geometry_tolerance default unit is in kilometers?

    if nodes and not edges:
        return node_gdf_from_graph(G, crs=crs)
    elif not nodes and edges:
        return edge_gdf_from_graph(G, crs=crs, smooth_geometry=smooth_geometry,
                                   smooth_geometry_tolerance=smooth_geometry_tolerance)

    return node_gdf_from_graph(G), edge_gdf_from_graph(G, crs=crs, smooth_geometry=smooth_geometry,
                                                       smooth_geometry_tolerance=smooth_geometry_tolerance)


def node_gdf_from_graph(G: nx.Graph, crs: str = 'WGS84'):
    G = G.copy()

    nodes, data = zip(*G.nodes(data=True))
    if all(['x' in d.keys() for d in data]) and all(['y' in d.keys() for d in data]):
        node_geometry = [Point(d['x'], d['y']) for d in data]
        gdf_nodes = gpd.GeoDataFrame(list(data), index=nodes, geometry=node_geometry)
        if "crs" in G.graph.keys():
            gdf_nodes.crs = G.graph["crs"]
    else:
        gdf_nodes = gpd.GeoDataFrame(list(data), index=nodes)

    return project_nodes_gdf(nodes_gdf=gdf_nodes, to_crs=crs)


def edge_gdf_from_graph(G: nx.Graph, crs: str = 'WGS84', smooth_geometry=True, smooth_geometry_tolerance=0.05):
    # smooth_geometry_tolerance default unit is in kilometers?

    G = G.copy()

    for e in G.edges():
        u = e[0]
        v = e[1]
        G.edges[u, v]['u'] = u
        G.edges[u, v]['v'] = v

    if G.number_of_edges() != 0:
        _, _, edge_data = zip(*G.edges(data=True))
        gdf_edges = gpd.GeoDataFrame(list(edge_data))
        gdf_edges = gdf_edges.groupby(by=['u', 'v'], as_index=True).first()
        if 'geometry' in edge_data[0].keys() and 'crs' in G.graph.keys():
            gdf_edges.crs = G.graph['crs']
    else:
        gdf_edges = gpd.GeoDataFrame()

    if smooth_geometry and 'geometry' in gdf_edges.columns:
        gdf_edges['geometry'] = gdf_edges['geometry'].simplify(smooth_geometry_tolerance)

    return project_edges_gdf(edges_gdf=gdf_edges, to_crs=crs)


'''
LOAD SUPPLEMENTAL DATA
'''


def load_splc_station_master():

    filename = 'SPLC_station_master.csv'
    filepath = os.path.join(GEN_DIR, filename)
    if not os.path.exists(filepath):
        # load geodataframes from geojson files on nodes in network
        df_n = gpd.read_file(os.path.join(NX_DIR, 'Routable_Rail_Lines_Only_Nodes.geojson'), header=0)  # nodes
        df_n = df_n[df_n['splc'] != ' ']
        df_n['splc'] = df_n['splc'].apply(lambda x: str(x) if len(str(x)) > 5 else '0' + str(x))
        df_n.rename(columns={'splc': 'SPLC'}, inplace=True)
        df_n.drop(columns=set(df_n.columns).difference({'SPLC', 'lon', 'lat'}), inplace=True)
        df_n.index = df_n['SPLC']
        df_n.drop_duplicates(subset=['SPLC'], keep='last', inplace=True)

        # load supplemental station list dataframes
        df_s = pd.read_csv(os.path.join(GEN_DIR, 'BNSF_station_list.csv'), header=0)
        df_s = df_s[df_s['County'] != 'CANADA']
        df_s.dropna(inplace=True)
        df_s['SPLC'] = df_s['SPLC'].astype(int)
        df_s['SPLC'] = df_s['SPLC'].apply(lambda x: str(x) if len(str(x)) > 5 else '0' + str(x))
        df_s.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)
        df_s.index = df_s['SPLC']
        df_s.drop_duplicates(subset=['SPLC'], keep='last', inplace=True)

        for splc in df_s.index:
            df_n.loc[splc, ['SPLC', 'lat', 'lon']] = df_s.loc[splc, ['SPLC', 'lat', 'lon']]

        df_n.to_csv(filepath, index=False)
    else:
        df_n = pd.read_csv(filepath, header=0)
        df_n['SPLC'] = df_n['SPLC'].apply(lambda x: str(x) if len(str(x)) > 5 else '0' + str(x))

    return df_n


def load_us_cities():

    filename = 'uscities.csv'
    filepath = os.path.join(GEN_DIR, filename)

    return pd.read_csv(filepath, header=0)


'''
PLOT
'''


def plot_states_bg(crs: None = 'EPSG:5070'):
    # load states gdf
    states_path = os.path.join(GEN_DIR, 'cb_2018_us_state_500k')
    states_df = gpd.read_file(states_path)
    # extract continguous states
    states_df = states_df[(states_df['STATEFP'].astype(int) < 60) & (states_df['STATEFP'].astype(int) != 15)
                          & (states_df['STATEFP'].astype(int) != 2)].copy()
    states_df.to_crs(crs=crs, inplace=True)
    # plot
    fig, ax = plt.subplots(figsize=(9, 6))
    states_df.boundary.plot(ax=ax, alpha=0.5, linewidth=0.5)
    # remove tick marks
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_title('Contiguous U.S.')

    return fig, ax


def plot_graph(G: nx.Graph, ax=None, nodes=None, edges=None, node_kwds=None, edge_kwds=None,
               plot_nodes=True, plot_edges=True, connected_only=False, title: str = None, crs: str = 'EPSG:5070'):
    # issues with CRS and plotting flow graph on CRS 'EPSG:5070'

    G = project_graph(G.copy(), to_crs=crs)

    if nodes is None:
        nodes = list(G.nodes())
    if edges is None:
        edges = list(G.edges())

    if node_kwds is None:
        node_kwds = dict()
    if edge_kwds is None:
        edge_kwds = dict()

    edges = edges + [(e[1], e[0]) for e in edges]
    nodes_to_remove = set(G.nodes()).difference(set(nodes))
    edges_to_remove = set(G.edges()).difference(set(edges))
    G = remove_from_graph(G, nodes_to_remove=nodes_to_remove, edges_to_remove=edges_to_remove,
                          connected_only=connected_only)

    # load gdfs for graph
    nodes_gdf, edges_gdf = gdfs_from_graph(G)
    # compose plotting arguments
    node_kwds_default = {'color': 'purple',
                         'marker': 'o',
                         'markersize': 30,
                         'zorder': 4}
    edge_kwds_default = {'color': 'green',
                         'linewidth': 1,
                         'zorder': 3}
    node_kwds.update({key: value for key, value in node_kwds_default.items() if key not in node_kwds.keys()})
    edge_kwds.update({key: value for key, value in edge_kwds_default.items() if key not in edge_kwds.keys()})

    # for special linewidth plotting
    custom_legend = False
    lw = edge_kwds['linewidth']
    if isinstance(lw, tuple):
        # when lw is a tuple, lw has the form (<edge_attr_name>, <max_linewidth>)
        # if <edge_attr_name> in columns of edge_gdf
        if lw[0] in edges_gdf.columns:
            # then we normalize this column and scale by <max_linewidth>
            units = ' Tons'
            if 'emissions' in lw[0]:
                units = ' tonne-CO$_2$e/mi'
            custom_legend = True
            e_lw = edges_gdf[lw[0]].values
            mx_e = max(e_lw)
            mn_e = min(e_lw)
            edges_gdf['linewidth'] = lw[1] * (e_lw - mn_e) / (mx_e - mn_e)
            edge_kwds['linewidth'] = edges_gdf['linewidth']

            legend_lines = [Line2D([0], [0], lw=i, color=edge_kwds['color']) for i in range(1, lw[1] + 1)]
            legend_names = [vals_str(mn_e + (mx_e - mn_e) * i / lw[1]) + units for i in range(1, lw[1] + 1)]
            # legend_names = [str(round(np.percentile(e_lw, q=(100 * i/lw[1])))) + ' Tons' for i in range(1, lw[1] + 1)]
        else:
            edge_kwds['linewidth'] = edge_kwds_default['linewidth']

    if ax is None:
        # plot states background
        fig, ax = plot_states_bg(crs=crs)
    # plot graph components
    if plot_nodes:
        nodes_gdf.plot(ax=ax, **node_kwds)
    if plot_edges:
        edges_gdf.plot(ax=ax, **edge_kwds)
        if custom_legend:
            ax.legend(legend_lines, legend_names)

    ax.set_title(title)

    return ax
