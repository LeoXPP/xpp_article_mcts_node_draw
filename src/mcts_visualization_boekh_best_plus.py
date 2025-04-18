import json
import math
import os
import glob
import numpy as np  # 引入 NumPy 进行矩阵运算

from bokeh.plotting import figure, curdoc
from bokeh.models import (
    ColumnDataSource,
    Slider,
    HoverTool,
    Button,
    Select,
    TextInput,
    Div,
    TapTool,
    MultiSelect,
    RangeSlider,
    CustomJS
)
from bokeh.layouts import column, row
from bokeh.palettes import Category10
from collections import defaultdict

def load_json_files(directory, pattern="xiepanpan_mcst_tree_*.json"):
    """
    加载指定目录下所有匹配模式的 JSON 文件，并按文件名排序。
    """
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    all_data = []
    for file in files:
        with open(file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"无法解码 JSON 文件: {file}")
                continue
            all_data.append((file, data))
    return all_data

def assign_colors(all_data):
    """
    为每个车辆分配颜色。自车（'ego'）使用红色，其他车辆使用 Category10 调色板中的颜色。
    """
    vehicle_ids = sorted(set(node["vehicle_id"] for _, data in all_data for node in data["nodes"]))
    palette = Category10[10]  # 使用 Category10 调色板的前10种颜色
    other_colors = list(palette)
    vehicle_colors = {}
    color_idx = 0
    for vid in vehicle_ids:
        if vid == "ego":
            vehicle_colors[vid] = (161, 199, 222)  # Self vehicle color (R 161, G 199, B 222)
        else:
            vehicle_colors[vid] = (245, 176, 176)  # Other vehicles color (R 245, G 176, B 176)

    return vehicle_colors

def get_plot_limits(nodes, scale=1.0):
    """
    根据节点的局部坐标计算绘图的范围，并添加一定的填充。
    """
    all_x = [node["x_local"] / scale for node in nodes]
    all_y = [node["y_local"] / scale for node in nodes]
    if not all_x or not all_y:
        # 默认范围
        return -10, 10, -10, 10
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    padding_x = 4.0 
    padding_y = 4.0 
    return min_x - padding_x, max_x + padding_x, min_y - padding_y, max_y + padding_y

def calculate_rotated_rectangle(x, y, length, width, angle_deg):
    """
    计算旋转后的矩形四个角的坐标。
    """
    theta = math.radians(angle_deg)
    dx = length / 2
    dy = width / 2
    corners = [
        (-dx, -dy),
        (-dx, dy),
        (dx, dy),
        (dx, -dy)
    ]
    rotated_corners = []
    for cx, cy in corners:
        rx = cx * math.cos(theta) - cy * math.sin(theta) + x
        ry = cx * math.sin(theta) + cy * math.cos(theta) + y
        rotated_corners.append((rx, ry))
    return rotated_corners

def point_in_polygon(x, y, poly_xs, poly_ys):
    """
    判断点 (x, y) 是否在多边形内部。
    """
    num = len(poly_xs)
    j = num - 1
    c = False
    for i in range(num):
        if ((poly_ys[i] > y) != (poly_ys[j] > y)) and \
           (x < (poly_xs[j] - poly_xs[i]) * (y - poly_ys[i]) / (poly_ys[j] - poly_ys[i] + 1e-9) + poly_xs[i]):
            c = not c
        j = i
    return c

def get_transform_matrix_from_initial_pose(initial_pose):
    """
    构建从世界坐标系到初始局部坐标系的变换矩阵。
    初始局部坐标系的x轴为横向，y轴为前进方向。
    """
    x0, y0, theta0 = initial_pose
    phi = math.pi / 2 - theta0  # 旋转角度，确保前进方向为正Y轴
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    # 正确的变换矩阵，包括旋转和平移
    T = np.array(
        [
            [cos_phi, -sin_phi, -x0 * cos_phi + y0 * sin_phi],
            [sin_phi, cos_phi, -x0 * sin_phi - y0 * cos_phi],
            [0, 0, 1]
        ]
    )
    return T

def apply_transformation(x, y, T):
    """
    应用变换矩阵 T 到点 (x, y)。
    """
    point = np.array([x, y, 1])
    transformed_point = T @ point
    return transformed_point[0], transformed_point[1]

def prepare_frame_data(data, vehicle_colors, scale=1.0, mode='Top N', top_n=3, node_id_filter=None):
    """
    为当前帧准备绘图数据，将全局坐标转换为局部坐标，并应用各种过滤器。
    当检测到 x 或 y 为 null 时，将跳过该节点，不进行绘制。
    """
    nodes = data.get("nodes", [])
    if not nodes:
        return None, None, None, None, None

    # 提取当前帧的初始ego车辆的位置信息
    initial_ego_node = next((node for node in nodes if node["vehicle_id"] == "ego"), None)
    if initial_ego_node is None:
        print("当前帧的ego车辆未找到。")
        return None, None, None, None, None

    initial_x0 = initial_ego_node["x"]
    initial_y0 = initial_ego_node["y"]
    initial_theta0 = initial_ego_node["theta"]

    # 构建变换矩阵
    T = get_transform_matrix_from_initial_pose((initial_x0, initial_y0, initial_theta0))

    # 过滤掉 x 或 y 为 None 的节点，并将全局坐标转换为局部坐标
    valid_nodes = []
    for node in nodes:
        # 如果 x 或 y 为 None，则跳过此节点
        if node.get("x") is None or node.get("y") is None:
            continue

        x_global = node["x"]
        y_global = node["y"]
        theta_global = node["theta"]

        x_local, y_local = apply_transformation(x_global, y_global, T)
        theta_local = theta_global - initial_theta0  # 调整方向角

        # 更新节点的局部坐标和方向角
        node["x_local"] = x_local
        node["y_local"] = y_local
        node["theta_local"] = theta_local

        valid_nodes.append(node)

    if not valid_nodes:
        print("所有节点均无效（x 或 y 为 null）。")
        return None, None, None, None, None

    # 根据局部坐标计算绘图范围
    min_x, max_x, min_y, max_y = get_plot_limits(valid_nodes, scale=scale)

    # 更新全局最大深度及过滤器（这里假设 max_depth_global、depth_range_filter、reward_range_filter 为全局变量）
    global max_depth_global
    max_depth = max(node["depth"] for node in valid_nodes)
    if max_depth > max_depth_global:
        max_depth_global = max_depth
        depth_range_filter.end = max_depth_global
        depth_range_filter.value = (depth_range_filter.value[0], max_depth_global)

    # 应用过滤器
    selected_nodes = [node for node in valid_nodes if
                      node["vehicle_id"] in vehicle_id_filter.value and
                      depth_range_filter.value[0] <= node["depth"] <= depth_range_filter.value[1] and
                      reward_range_filter.value[0] <= node.get("reward", 0) <= reward_range_filter.value[1]]

    # 应用节点 ID 筛选
    if node_id_filter:
        node_ids = [nid.strip() for nid in node_id_filter.split(',') if nid.strip()]
        if node_ids:
            selected_nodes = [node for node in selected_nodes if node.get('id', '') in node_ids]

    # 如果按照深度分组并选择 Top N，需要调整
    if mode == 'Top N':
        depth_groups = defaultdict(list)
        for node in selected_nodes:
            depth = node["depth"]
            depth_groups[depth].append(node)

        filtered_nodes = []
        for depth in sorted(depth_groups.keys()):
            group = depth_groups[depth]
            sorted_group = sorted(group, key=lambda n: n.get("reward", 0), reverse=True)
            top_n_nodes = sorted_group[:top_n]
            filtered_nodes.extend(top_n_nodes)
        selected_nodes = filtered_nodes

    polygons = []
    for node in selected_nodes:
        # x_local = node["x_local"] / scale
        # y_local = node["y_local"] / scale
        # # 变换坐标：旋转90度，(x, y) -> (y, -x)
        # new_x = y_local
        # new_y = -x_local
        # theta = node["theta_local"]
        # # 调整方向角，减去90度
        # new_theta = (theta - 90) % 360
        
        
        x_local = node["x_local"] / scale
        y_local = node["y_local"] / scale
        # 变换坐标：旋转90度，(x, y) -> (y, -x)
        new_x = y_local
        new_y = -x_local
        theta = node["theta_local"] - math.pi / 2  # 新角度减去90度


        # 车辆尺寸：长1米，宽2米（根据您提供的注释调整）
        # length = 0.5
        # width = 0.5
        length = 1.8
        width = 4.5

        # 计算旋转后的矩形四个角的坐标，使用变换后的坐标和角度
        # corners = calculate_rotated_rectangle(new_x, new_y, length, width, new_theta)
        corners = calculate_rotated_rectangle(new_x, new_y, length, width, math.degrees(theta))
        
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]

        # 获取 vehicle_rewards，处理可能为空的情况
        vehicle_rewards = node.get("vehicle_rewards", {})
        vehicle_rewards_serialized = json.dumps(vehicle_rewards, indent=2)

        # 获取 vehicle_states，处理可能为空的情况
        vehicle_states = node.get("vehicle_states", {})
        vehicle_states_serialized = json.dumps(vehicle_states, indent=2)

        # 获取节点相关信息
        vid = node["vehicle_id"]
        depth = node["depth"]
        reward = node.get("reward", 0)
        color = vehicle_colors.get(vid, 'blue')
        alpha = 1.0 - (depth / (max_depth + 1))
        visits = node.get("visits", 0)
        expanded_num = node.get("expanded_num", 0)
        iter_ = node.get("iter", 0)
        id_ = node.get("id", "")
        size = node.get("size", 0)
        max_size = node.get("max_size", 0)
        is_valid = node.get("is_valid", False)
        static_reward = node.get("static_reward", 0.0)
        relative_time = node.get("relative_time", 0.0)
        acc = node.get("acc", 0.0)  # 添加 acc
        vel = node.get("vel", 0.0)  # 添加 vel

        # Determine category for legend
        category = 'ego' if vid == 'ego' else 'other'

        polygons.append({
            'xs': xs,
            'ys': ys,
            'color': color,
            'alpha': alpha,
            'category': category,  # 添加 category
            'vid': vid,
            'depth': depth,
            'reward': reward,
            'acc': acc,  # 添加 acc
            'vel': vel,  # 添加 vel
            'vehicle_rewards': vehicle_rewards_serialized,
            'vehicle_states': vehicle_states_serialized,  # 添加 vehicle_states
            'visits': visits,
            'expanded_num': expanded_num,
            'iter': iter_,
            'id': id_,
            'size': size,
            'max_size': max_size,
            'is_valid': is_valid,
            'static_reward': static_reward,
            'relative_time': relative_time,
        })

    # 创建数据字典
    data_dict = dict(
        xs=[poly['xs'] for poly in polygons],
        ys=[poly['ys'] for poly in polygons],
        color=[poly['color'] for poly in polygons],
        alpha=[poly['alpha'] for poly in polygons],
        category=[poly['category'] for poly in polygons],
        vid=[poly['vid'] for poly in polygons],
        depth=[poly['depth'] for poly in polygons],
        reward=[poly['reward'] for poly in polygons],
        acc=[poly['acc'] for poly in polygons],
        vel=[poly['vel'] for poly in polygons],
        vehicle_rewards=[poly['vehicle_rewards'] for poly in polygons],
        vehicle_states=[poly['vehicle_states'] for poly in polygons],
        visits=[poly['visits'] for poly in polygons],
        expanded_num=[poly['expanded_num'] for poly in polygons],
        iter=[poly['iter'] for poly in polygons],
        id=[poly['id'] for poly in polygons],
        size=[poly['size'] for poly in polygons],
        max_size=[poly['max_size'] for poly in polygons],
        is_valid=[poly['is_valid'] for poly in polygons],
        static_reward=[poly['static_reward'] for poly in polygons],
        relative_time=[poly['relative_time'] for poly in polygons],
    )

    current_vehicle_ids = sorted(set(node["vehicle_id"] for node in selected_nodes))

    return data_dict, (min_y, max_y, -max_x, -min_x), current_vehicle_ids, max_depth, selected_nodes

def update_plot(attr, old, new):
    """
    更新绘图数据和视图范围。
    """
    frame_idx = slider.value
    file, data = all_data[frame_idx]

    current_mode = mode_select.value
    current_top_n = top_n_slider.value if current_mode == 'Top N' else None
    node_id_filter = node_id_filter_input.value  # 获取节点 ID 过滤器的值

    new_data, plot_limits, current_vehicle_ids, max_depth, nodes = prepare_frame_data(
        data,
        vehicle_colors,
        scale=scale,
        mode=current_mode,
        top_n=top_n_slider.value,
        node_id_filter=node_id_filter  # 传递节点 ID 过滤器
    )

    if new_data is None:
        return

    p.x_range.start = plot_limits[0]
    p.x_range.end = plot_limits[1]
    p.y_range.start = plot_limits[2]
    p.y_range.end = plot_limits[3]

    source.data = new_data
    p.title.text = f'MCTS Tree Vehicle Visualization\nFile: {os.path.basename(file)}'

    # Update road lines based on new plot limits
    min_y, max_y = plot_limits[0], plot_limits[1]

    main_centerline_source.data = dict(
        x=[min_y, max_y],
        y=[0, 0]
    )

    shifted_centerline_source.data = dict(
        x=[min_y, max_y],
        y=[4, 4]
    )

    boundary_right_source.data = dict(
        x=[min_y, max_y],
        y=[-2, -2]
    )

    boundary_left_source.data = dict(
        x=[min_y, max_y],
        y=[6, 6]
    )

    # Update road fill
    road_fill_source.data = dict(
        x=[min_y, max_y, max_y, min_y],
        y=[-2, -2, 6, 6]
    )

def update_mode(attr, old, new):
    """
    更新显示模式（Top N 或 All）。
    """
    if mode_select.value == 'Top N':
        top_n_slider.disabled = False
    else:
        top_n_slider.disabled = True
    update_plot(None, None, None)

def update_top_n(attr, old, new):
    """
    更新 Top N 滑块的值。
    """
    update_plot(None, None, None)

def update_plot_dimensions(attr, old, new):
    """
    动态调整绘图的宽度和高度。
    """
    p.width = width_slider.value
    p.height = height_slider.value

def main():
    global all_data, vehicle_colors, scale, p, slider, source, hover, mode_select, top_n_slider
    global vehicle_id_filter, depth_range_filter, reward_range_filter, max_depth_global
    global node_id_filter_input
    global width_slider, height_slider  # 添加宽度和高度滑块变量
    global main_centerline_source, shifted_centerline_source
    global boundary_right_source, boundary_left_source, road_fill_source

    directory = '.'   # JSON 文件目录
    scale = 1.0

    all_data = load_json_files(directory, pattern="../data/xiepanpan_mcst_tree_*.json")
    if not all_data:
        print("OH MO !!! JSON NOT FOUND")
        return

    vehicle_colors = assign_colors(all_data)

    # 初始化全局最大深度
    max_depth_global = max(node["depth"] for _, data in all_data for node in data["nodes"] if node.get("depth") is not None)

    # 车辆 ID 过滤器
    vehicle_ids = sorted(set(node["vehicle_id"] for _, data in all_data for node in data["nodes"]))
    vehicle_id_filter = MultiSelect(title="Filter by Vehicle ID", value=vehicle_ids, options=vehicle_ids)
    vehicle_id_filter.on_change('value', lambda attr, old, new: update_plot(None, None, None))

    # 深度范围过滤器
    depth_range_filter = RangeSlider(start=0, end=max_depth_global, value=(0, max_depth_global), step=1, title="Filter by Depth")
    depth_range_filter.on_change('value', lambda attr, old, new: update_plot(None, None, None))

    # 奖励值范围过滤器
    min_reward = min(node.get("reward", 0) for _, data in all_data for node in data["nodes"])
    max_reward = max(node.get("reward", 0) for _, data in all_data for node in data["nodes"])
    reward_range_filter = RangeSlider(start=min_reward, end=max_reward, value=(min_reward, max_reward), step=0.1, title="Filter by Reward")
    reward_range_filter.on_change('value', lambda attr, old, new: update_plot(None, None, None))

    # 节点 ID 过滤器
    node_id_filter_input = TextInput(title="Filter by Node ID (comma-separated)", value="")
    node_id_filter_input.on_change('value', lambda attr, old, new: update_plot(None, None, None))

    initial_frame_idx = 0
    file, data = all_data[initial_frame_idx]
    new_data, plot_limits, current_vehicle_ids, max_depth, nodes = prepare_frame_data(
        data,
        vehicle_colors,
        scale=scale,
        mode='Top N',
        top_n=3,
        node_id_filter=""  # 初始无过滤
    )

    if new_data is None:
        print("初始帧数据准备失败。")
        return

    # 创建 ColumnDataSource
    source = ColumnDataSource(data=new_data)

    # 创建绘图，确保使用 canvas 后端
    p = figure(
        title=f'MCTS Tree Vehicle Visualization\nFile: {os.path.basename(file)}',
        x_range=(plot_limits[0], plot_limits[1]),
        y_range=(plot_limits[2], plot_limits[3]),
        width=800,
        height=800,  # 初始高度
        tools="pan,box_zoom,reset,save",  # 添加 save 工具
        output_backend="canvas"  # 确保使用 canvas 后端
    )
    p.background_fill_color = "white"  # 设置背景为白色
    p.grid.grid_line_color = None  # 去除网格线
    p.xaxis.axis_label = ' Y Position (m)'
    p.yaxis.axis_label = ' X Position (m)'
    p.title.text_font_size = '30pt'
    p.xaxis.axis_label_text_font_size = '20pt'
    p.yaxis.axis_label_text_font_size = '20pt'

    # Set all fonts to Times New Roman
    p.title.text_font = 'times'  # Font for title
    p.xaxis.axis_label_text_font = 'times'  # Font for x-axis labels
    p.yaxis.axis_label_text_font = 'times'  # Font for y-axis labels
    p.xaxis.major_label_text_font = 'times'  # Font for x-axis ticks
    p.yaxis.major_label_text_font = 'times'  # Font for y-axis ticks

    # Modify tick labels' font size
    p.xaxis.major_label_text_font_size = '30pt'  # Font size for x-axis ticks
    p.yaxis.major_label_text_font_size = '30pt'  # Font size for y-axis ticks

    # p.legend.label_text_font_size 和 p.legend.draggable 将在添加图例后设置

    # 创建 ColumnDataSources for road lines and fill
    min_y, max_y = plot_limits[0], plot_limits[1]

    main_centerline_source = ColumnDataSource(data=dict(
        x=[min_y, max_y],
        y=[0, 0]
    ))

    shifted_centerline_source = ColumnDataSource(data=dict(
        x=[min_y, max_y],
        y=[4, 4]
    ))

    boundary_right_source = ColumnDataSource(data=dict(
        x=[min_y, max_y],
        y=[-4, -4]
    ))

    boundary_left_source = ColumnDataSource(data=dict(
        x=[min_y, max_y],
        y=[6, 6]
    ))

    road_fill_source = ColumnDataSource(data=dict(
        x=[min_y, max_y, max_y, min_y],
        y=[-4, -4, 6, 6]
    ))

    # 添加道路填充
    road_fill_renderer = p.patch(
        'x', 'y',
        source=road_fill_source,
        fill_color='white',
        fill_alpha=0.5,
        line_alpha=0
    )

    # 添加道路中心线和边界线
    main_centerline_renderer = p.line(
        'x', 'y',
        source=main_centerline_source,
        line_width=2,
        line_dash='dashed',
        line_color='black'
    )

    shifted_centerline_renderer = p.line(
        'x', 'y',
        source=shifted_centerline_source,
        line_width=2,
        line_dash='dashed',
        line_color='black'
    )

    boundary_right_renderer = p.line(
        'x', 'y',
        source=boundary_right_source,
        line_width=2,
        line_color='black'
    )

    boundary_left_renderer = p.line(
        'x', 'y',
        source=boundary_left_source,
        line_width=2,
        line_color='black'
    )

    # 创建车辆多边形渲染器
    renderer = p.patches(
        'xs', 'ys',
        source=source,
        fill_color='color',
        fill_alpha='alpha',
        line_color='black',
        line_width=2,
        legend_field='category'  # 使用 category 作为 legend_field
    )

    # 添加 TapTool
    taptool = TapTool()
    p.add_tools(taptool)

    # 奖励详情显示区域
    reward_details_div = Div(text="<h2>Vehicle Rewards Details</h2><p>Click on a vehicle to see details.</p>", width=400)

    # 点击事件回调函数
    def vehicle_tap_callback(event):
        # 获取点击的位置
        x = event.x
        y = event.y

        # 在 source 中查找包含点击点的车辆
        indices = []
        for i in range(len(source.data['xs'])):
            xs = source.data['xs'][i]
            ys = source.data['ys'][i]
            if point_in_polygon(x, y, xs, ys):
                indices.append(i)

        if indices:
            # 假设只取第一个匹配的车辆
            index = indices[0]
            vid = source.data['vid'][index]
            vehicle_rewards = source.data['vehicle_rewards'][index]
            vehicle_states = source.data['vehicle_states'][index]
            depth = source.data['depth'][index]
            reward = source.data['reward'][index]
            acc = source.data['acc'][index]        # 获取 acc
            vel = source.data['vel'][index]        # 获取 vel
            visits = source.data['visits'][index]
            expanded_num = source.data['expanded_num'][index]
            iter_ = source.data['iter'][index]
            id_ = source.data['id'][index]
            size = source.data['size'][index]
            max_size = source.data['max_size'][index]
            is_valid = source.data['is_valid'][index]
            static_reward = source.data['static_reward'][index]
            relative_time = source.data['relative_time'][index]

            # 格式化 vehicle_rewards
            vehicle_rewards_dict = json.loads(vehicle_rewards)
            rewards_html = "<ul>"
            for key, value in vehicle_rewards_dict.items():
                rewards_html += f"<li><b>{key}:</b> {value}</li>"
            rewards_html += "</ul>"

            # 格式化 vehicle_states
            vehicle_states_dict = json.loads(vehicle_states)
            vehicle_states_html = "<ul>"
            for key, value in vehicle_states_dict.items():
                vehicle_states_html += f"<li><b>{key}:</b> {value}</li>"
            vehicle_states_html += "</ul>"

            # 更新节点信息显示
            node_info_html = f"""
            <h2>Vehicle ID: {vid}</h2>
            <p><b>ID:</b> {id_}</p>
            <p><b>Depth:</b> {depth}</p>
            <p><b>Reward:</b> {reward}</p>
            <p><b>Acceleration:</b> {acc}</p>
            <p><b>Velocity:</b> {vel}</p>
            <p><b>Visits:</b> {visits}</p>
            <p><b>Expanded Num:</b> {expanded_num}</p>
            <p><b>Iteration:</b> {iter_}</p>
            <p><b>Size:</b> {size}</p>
            <p><b>Max Size:</b> {max_size}</p>
            <p><b>Is Valid:</b> {is_valid}</p>
            <p><b>Static Reward:</b> {static_reward}</p>
            <p><b>Relative Time:</b> {relative_time}</p>
            <h3>Vehicle Rewards:</h3>
            {rewards_html}
            <h3>Vehicle States:</h3>
            {vehicle_states_html}
            """

            reward_details_div.text = node_info_html
        else:
            reward_details_div.text = "<h2>No vehicle selected</h2>"

    # 将回调函数绑定到点击事件
    p.on_event('tap', vehicle_tap_callback)

    # 添加 HoverTool
    hover = HoverTool(
        renderers=[renderer],
        tooltips=[
            ("Vehicle ID", "@vid"),
            ("Depth", "@depth"),
            ("Reward", "@reward"),
            ("Acceleration", "@acc"),  # 添加 Acceleration
            ("Velocity", "@vel"),       # 添加 Velocity
        ]
    )
    p.add_tools(hover)

    # 创建滑块和按钮
    slider = Slider(
        start=0,
        end=len(all_data)-1,
        value=0,
        step=1,
        title="Frame",
        width=300
    )
    slider.on_change('value', update_plot)

    mode_select = Select(
        title="Display Mode",
        value="Top N",
        options=["Top N", "All"],
        width=300
    )
    mode_select.on_change('value', update_mode)

    top_n_slider = Slider(
        start=1,
        end=10,
        value=3,
        step=1,
        title="Top N per Depth",
        disabled=False,
        width=300
    )
    top_n_slider.on_change('value', update_top_n)

    play_button = Button(label="► Play", width=100)
    is_playing = False

    def play():
        nonlocal is_playing
        is_playing = True
        play_button.label = "❚❚ Pause"
        curdoc().add_periodic_callback(callback, 1000)  # 1 frame per sec

    def pause():
        nonlocal is_playing
        is_playing = False
        play_button.label = "► Play"
        try:
            curdoc().remove_periodic_callback(callback)
        except ValueError:
            pass

    def callback():
        new_value = (slider.value + 1) % len(all_data)
        slider.value = new_value

    def toggle_play():
        nonlocal is_playing
        if is_playing:
            pause()
        else:
            play()

    play_button.on_click(toggle_play)

    # 添加动态调整图像横纵比例的滑块
    width_slider = Slider(
        start=300,
        end=3200,
        value=2650,
        step=50,
        title="Plot Width (pixels)",
        width=300
    )
    width_slider.on_change('value', update_plot_dimensions)

    height_slider = Slider(
        start=300,
        end=3200,
        value=700,
        step=50,
        title="Plot Height (pixels)",
        width=300
    )
    height_slider.on_change('value', update_plot_dimensions)

    # 创建“一键保存图像”按钮
    save_button = Button(label="Save Image", button_type="success", width=100)

    # 定义 CustomJS 回调，用于保存图像
    save_callback = CustomJS(args=dict(p=p), code="""
        // Access the plot view
        const plot_view = p.view;
        if (!plot_view) {
            console.log("Plot view not found.");
            return;
        }

        // Access the canvas
        const canvas = plot_view.canvas_view.canvas;
        if (!canvas) {
            console.log("Canvas not found.");
            return;
        }

        // Convert the canvas to a data URL
        const dataURL = canvas.toDataURL("image/png");

        // Create a temporary link to trigger the download
        const link = document.createElement('a');
        link.href = dataURL;
        link.download = "plot.png";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    """)

    # 将回调绑定到按钮点击事件
    save_button.js_on_click(save_callback)

    # 将控件放在左侧，包括新的节点 ID 过滤器、图像比例滑块和保存按钮
    controls = column(
        slider,
        mode_select,
        top_n_slider,
        play_button,
        save_button,            # 添加保存按钮
        vehicle_id_filter,
        depth_range_filter,
        reward_range_filter,
        node_id_filter_input,  # 添加节点 ID 过滤器到控件
        width_slider,          # 添加图像宽度滑块
        height_slider          # 添加图像高度滑块
    )
    left_side = controls

    # 设置控件宽度
    for control in controls.children:
        control.width = 300

    # 设置绘图区域宽度和高度
    p.width = width_slider.value
    p.height = height_slider.value

    # 设置奖励详情显示区域宽度
    reward_details_div.width = 400

    # Final layout
    layout = row(left_side, p, reward_details_div)

    # 添加图例属性设置（确保图例已添加）
    def set_legend_properties():
        if p.legend:
            for legend in p.legend:
                legend.label_text_font_size = '12pt'

    # 调用一次以确保图例属性被设置
    set_legend_properties()

    curdoc().add_root(layout)
    curdoc().title = "MCTS Tree Vehicle Visualization"
main()