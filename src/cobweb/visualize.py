"""
The visualize module provides functions for generating html visualizations of
trees created by the other modules of concept_formation.
"""

import webbrowser
from pathlib import Path
from shutil import copy

from cobweb.cobweb_discrete import CobwebNode


def _copy_file(filename, target_dir):
    module_path = Path(__file__).parent
    src = module_path / "visualization_files" / filename
    dst = Path(target_dir) / filename
    copy(src, dst)
    return str(dst)


def _gen_output_file(js_ob):
    return "(function (){ window.trestle_output=" + js_ob + "; })();"


def _gen_viz(js_ob, dst, recreate_html):
    if dst is None:
        module_path = Path(__file__).parent
        output_file = module_path / "visualization_files" / "output.js"
        with output_file.open("w") as out:
            out.write(_gen_output_file(js_ob))
        viz_html_file = module_path / "visualization_files" / "viz.html"
        webbrowser.open(f"file://{viz_html_file.resolve()}")
    else:
        dst_path = Path(dst)
        viz_html_path = dst_path / "viz.html"
        if recreate_html or not viz_html_path.exists():
            viz_file = _copy_file("viz.html", dst)
            _copy_file("viz_logic.js", dst)
            _copy_file("viz_styling.css", dst)
        else:
            viz_file = viz_html_path
        with (dst_path / "output.js").open("w") as out:
            out.write(_gen_output_file(js_ob))
        webbrowser.open(f"file://{viz_file.resolve()}")


def visualize(tree, dst=None, recreate_html=True):
    """
    Create an interactive visualization of a concept_formation tree and open
    it in your browswer.

    If a destination directory is specified this function will create html,
    js, and css files in the destination directory provided. By default this
    will always recreate the support html, js, and css files but a flag can
    turn this off.

    :param tree: A category tree to visualize
    :param dst: A directory to generate visualization files into. If None no
        files will be generated
    :param create_html: A flag for whether new supporting html files should be
        created
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`,
        :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or
        :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :type dst: str
    :type create_html: bool
    """
    _gen_viz(tree.root.output_json(), dst, recreate_html)


# begin Modification
# (for saving & loading the tree)
def save_tree(tree, id):
    js_ob = tree.dump_json()
    filename = f"output_{id}.js"
    module_path = Path(__file__).parent
    output_file = module_path / "saved_cobweb_trees" / filename
    with output_file.open("w") as json_file:
        json_file.write(js_ob)


def load_tree(tree, id):
    filename = f"output_{id}.js"
    module_path = Path(__file__).parent
    output_file = module_path / "saved_cobweb_trees" / filename
    with output_file.open() as js_file:
        js_code = js_file.read()

    tree.load_json(js_code)
    return tree


# end Modification


def _trim_leaves(j_ob):
    ret = {k: j_ob[k] for k in j_ob if k != "children"}
    ret["children"] = [
        _trim_leaves(child) for child in j_ob["children"] if len(child["children"]) > 0
    ]
    return ret


def visualize_no_leaves(tree, cuts=1, dst=None, recreate_html=True):
    """
    Create an interactive visualization of a concept_formation tree cuts levels
    above the leaves and open it in your browswer.

    This visualization differs from the normal one by trimming the leaves from
    the tree. This is often useful in seeing patterns when the individual
    leaves are overly frequent visual noise.

    If a destination directory is specified this function will create html,
    js, and css files in the destination directory provided. By default this
    will always recreate the support html, js, and css files but a flag can
    turn this off.

    :param tree: A category tree to visualize
    :param cuts: The number of times to trim up the leaves
    :param dst: A directory to generate visualization files into. If None no
        files will be generated
    :param create_html: A flag for whether new supporting html files should be
        created
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`,
        :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or
        :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :type cuts: int
    :type dst: str
    :type create_html: bool
    """
    j_ob = tree.root.output_json()
    for _ in range(cuts):
        j_ob = _trim_leaves(j_ob)
    _gen_viz(j_ob, dst, recreate_html)


def _trim_to_clusters(j_ob, clusters):
    ret = {k: j_ob[k] for k in j_ob if k != "children"}
    if j_ob["name"] not in clusters:
        ret["children"] = [
            _trim_to_clusters(child, clusters) for child in j_ob["children"]
        ]
    else:
        ret["children"] = []
    return ret


def visualize_clusters(tree, clusters, dst=None, recreate_html=True):
    """
    Create an interactive visualization of a concept_formation tree trimmed to
    the level specified by a clustering from the cluster module.

    This visualization differs from the normal one by trimming the tree to the
    level of a clustering. Basically the output traverses down the tree but
    stops recursing if it hits a node in the clustering. Both label or concept
    based clusterings are supported as the relevant names will be extracted.

    If a destination directory is specified this function will create html,
    js, and css files in the destination directory provided. By default this
    will always recreate the support html, js, and css files but a flag can
    turn this off.

    :param tree: A category tree to visualize
    :param clusters: A list of cluster labels or concept nodes generated by
        the cluster module.
    :param dst: A directory to generate visualization files into. If None no
        files will be generated
    :param create_html: A flag for whether new supporting html files should be
        created
    :type tree: :class:`CobwebTree <concept_formation.cobweb.CobwebTree>`,
        :class:`Cobweb3Tree <concept_formation.cobweb3.Cobweb3Tree>`, or
        :class:`TrestleTree <concept_formation.trestle.TrestleTree>`
    :type clusters: list
    :type dst: str
    :type create_html: bool
    """
    if isinstance(clusters[0], CobwebNode):
        clusters = {str(c.concept_id) for c in clusters}
    else:
        clusters = set(clusters)

    j_ob = tree.root.output_json()
    j_ob = _trim_to_clusters(j_ob, clusters)
    _gen_viz(j_ob, dst, recreate_html)
