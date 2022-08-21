"""
    自动筛选、下载landsat8数据并解压指定波段文件
"""
import json
import os,sys
import tarfile

from landsatxplore.api import API
from landsatxplore.earthexplorer import EarthExplorer
import geopandas as gpd
import warnings
warnings.filterwarnings(action="ignore")

username = 'crook_wei'  # 输入EE账号
password = 'BwUP98ETwnCXVPq'  # 输入账号密码
# 初始化API接口获取key
api = API(username, password)

"""影像查询"""
def search_image(city,dataset,start_date,end_date,max_cloud_cover,bbox,stripes,years):
    # 输入要查询的边界
    #data = gpd.read_file(search_file)
    # 输入轨道矢量
    #if dataset.lower() == "sentinel_2a":
    #    grid_file = 'E:\*****\sentinel2_grid.shp'
    #elif dataset.lower() == "landsat_8_c1":
        # 输入landsat轨道矢量
    #    grid_file = 'E:\*****\WRS2_descending.shp'
    #wrs = gpd.GeoDataFrame.from_file(grid_file)
    # 查询边界覆盖的轨道中心坐标
    #wrs_intersection = wrs[wrs.intersects(data.geometry[0])]
    #longitude = (wrs_intersection.centroid).x.values
    #latitude = (wrs_intersection.centroid).y.values
    # print(longitude, latitude)
    for year in years:
        """查询"""
        year = str(year)
        scenes = api.search(
            dataset=dataset,    # 数据类型，如landsat_8_c1
            #latitude=float(latitude), # 坐标点
            #longitude=float(longitude),
            bbox= bbox, # 范围查询，全覆盖
            start_date=f"{year}-01-01", # 起始日期
            end_date=f"{year}-12-31",  # 终止日期
            max_cloud_cover=max_cloud_cover)    # 云量
        ids_old = [id['display_id'] for id in scenes]
        ids = []
        """根据stripes筛选条带号"""

        if stripes[0] !='0':
            for i in range(len(ids_old)):
                if ids_old[i].split("_")[2] in stripes:
                    ids.append(ids_old[i])
        else:
            ids = ids_old
        print("{} {}有{}幅图片云量低于{}".format(year,city, len(ids),cloud_cover))
        ids = {city:ids}
        if not os.path.exists(f"cloud{max_cloud_cover}_yaogan_fileter"):
            os.makedirs(f"cloud{max_cloud_cover}_yaogan_fileter")
        # 保存到文件
        filter = rf"cloud{max_cloud_cover}_yaogan_fileter\{city}_filter.json"
        old_data = {}
        if os.path.exists(filter):
            with open(filter, "r", encoding="utf-8") as f:
                file = f.read()
                if len(file) > 0:
                    old_data = json.loads(file)
                # 如果之前没有该城市，更新；有则添加
                if city not in old_data.keys():
                    old_data.update(ids)
                else:
                    old_data[city].extend(ids[city])
        with open(filter, "w", encoding="utf-8") as f:
            json.dump(old_data, f,indent=2)

    return scenes

# 下载影像数据
def Download_from_Landsatexplore(dataset,scene_list,run_text):
    """output_dir_city : 城市文件夹路径"""
    output_dir_city = ''
    if len(scene_list) > 0:
        # 根据ID下载影像
        for scene in scene_list:
            if dataset.lower() == "landsat_8_c1":
                output_dir = 'F:\文件\landsat8'# 输入下载路径
            elif dataset.lower() == "sentinel_2a":
                output_dir = 'F:\文件\Sentinel2'  # 输入下载路径
            output_dir_city = output_dir+'\\'+ run_text.split(" ")[0]
            exist_files = os.listdir(output_dir_city)
            if scene['display_id'] + ".tar.gz" in exist_files:
                print(scene['display_id'] + ".tar.gz" + " is existed.")
                continue
            if not os.path.isdir(output_dir_city):
                os.makedirs(output_dir_city)
            ee = EarthExplorer(username, password)
            print("Downloading: "+scene['display_id'])
            ee.download(identifier=scene['entity_id'], output_dir=output_dir_city)  #根据entity_id下载，etc.：LC08_L1GT_123038_20140531_20170422_01_T2
        ee.logout()
        """该次城市文件下载完成后立即解压到当前目录"""
        untar_city(output_dir_city)
        print(run_text.split(" ")[0] + " done")
        
        
def un_tar(file_name):
    # untar zip file
    tar = tarfile.open(file_name)
    old_names = tar.getnames()
    target_path = os.path.dirname(file_name) + "\\" + file_name.split("_")[3]
    if os.path.isdir(target_path):
        pass
    else:
        os.mkdir(target_path)
    # 只解压指定波段，按名称筛选
    targets = ['B2','B3','B4','B5']
    names = []
    for i in range(len(old_names)):
        if old_names[i].split("_")[-1][:2] in targets:
            names.append(old_names[i])
    for i in range(len(names)):
        if os.path.exists(target_path + "\\" + names[i]):
            print("{} is existed.".format(names[i]))
        else:
            print("{} is untaring...".format(names[i]))
            # 此时会有日期相同、条带号不同的文件被解压到同一文件夹下，后续需要进行合并切割
            tar.extract(names[i], target_path)
    tar.close()

"""文件格式：\\tar.gz压缩文件"""
def untar_city(city_path):
    old_files = os.listdir(city_path)
    files = []
    """筛选tar文件"""
    for i in range(len(old_files)):
        if old_files[i].split(".")[-1] == 'gz':
            files.append(old_files[i])
    for file in files:
        file_path = city_path + "\\" + file
        un_tar(file_path)

def run(text,cloud_cover):
    # 输入查询条件
    l = text.split(" ")
    city = l[0]
    dataset = l[1]#'landsat_8_c1' # 数据集
    start_date = l[2]#'2021-05-01' # 开始日期0
    end_date = l[3]#'2021-05-20' # 结束日期
    cloud_cover = cloud_cover#5 # 云量（%）
    # 输入查询文件
    #search_file = sys.argv[5]#r'E:\***\北京市.shp'  # 输入查询矢量
    #longitude = l[5]
    #latitude = l[6]
    b = l[5]
    bbox = tuple([float(i) for i in b.split(",")])
    s = l[6]
    stripes = [j for j in s.split(",")]
    # 查询数据
    years = [i for i in range(2013,2022)]
    search_list = search_image(city,dataset, start_date, end_date, cloud_cover, bbox,stripes,years)
    return search_list

if __name__ == '__main__':

    """Landsat 5 TM Collection 1 Level 1-->landsat_tm_c1
        Landsat 5 TM Collection 2 Level 1-->landsat_tm_c2_l1
        Landsat 5 TM Collection 2 Level 2-->landsat_tm_c2_l2
        Landsat 7 ETM+ Collection 1 Level 1-->landsat_etm_c1
        Landsat 7 ETM+ Collection 2 Level 1-->landsat_etm_c2_l1
        Landsat 7 ETM+ Collection 2 Level 2-->landsat_etm_c2_l2
        Landsat 8 Collection 1 Level 1-->landsat_8_c1
        Landsat 8 Collection 2 Level 1-->landsat_ot_c2_l1
        Landsat 8 Collection 2 Level 2-->landsat_ot_c2_l2
        Sentinel 2A-->sentinel_2a"""

    line_2013 = '{0} landsat_8_c1 2013-01-01 2018-01-01 5 {1},{4},{3},{2} 0'
    line_2018 = '{0} landsat_8_c1 2018-01-01 2022-01-01 5 {1},{4},{3},{2} 0'
    # 每次查询数量有限，每个城市分两批
    china_text = ['wuhan landsat_8_c1 2013-01-01 2018-01-01 5 113.6067,29.919,115.1897,31.4186 122039,122038,123039,123038','wuhan landsat_8_c1 2018-01-01 2022-01-01 5 113.6067,29.919,115.1897,31.4186 122039,122038,123039,123038',
                line_2013.format('guangzhou','112.8886','22.5304','114.1367','23.9022'),line_2018.format('guangzhou','112.8886','22.5304','114.1367','23.9022'),
                line_2013.format('shanghai','120.7543','30.6608','122.0582','31.9024'),line_2018.format('shanghai','120.7543','30.6608','122.0582','31.9024'),
                line_2013.format('nanjing','118.3117','31.1913','119.3290','32.6385'),line_2018.format('nanjing','118.3117','31.1913','119.3290','32.6385'),
                line_2013.format('hefei','116.5415','30.9071','118.0508','32.6002'),line_2018.format('hefei','116.5415','30.9071','118.0508','32.6002'),
                line_2013.format('xian','107.658347','33.696002','109.821747','34.743983'),line_2018.format('xian','107.658347','33.696002','109.821747','34.743983'),
                line_2013.format('chengdu','102.989623','30.090979','104.896262','31.437765'),line_2018.format('chengdu','102.989623','30.090979','104.896262','31.437765'),
                line_2013.format('changsha','111.890861','27.851024','114.256514','28.664368'),line_2018.format('changsha','111.890861','27.851024','114.256514','28.664368')
                ]

    

    europe_text  = [line_2013.format('beijing','116.0630','39.6824','116.7300','40.1869'),line_2018.format('beijing','116.0630','39.6824','116.7300','40.1869'),
                line_2013.format('berlin','13.0558','52.3375','13.7780','52.68'),line_2018.format('berlin','13.0558','52.3375','13.7780','52.68'),
                line_2013.format('budapest','18.9066','47.6088','19.3621','47.3349'),line_2018.format('budapest','18.9066','47.6088','19.3621','47.3349'),
                line_2013.format('london','-0.5910', '51.7323', '0.3318', '51.2425'),line_2018.format('london','-0.5910', '51.7323', '0.3318', '51.2425'),
                line_2013.format('riga','23.9088', '57.0973', '24.3437', '56.8499'),line_2018.format('riga','23.9088', '57.0973', '24.3437', '56.8499'),
                line_2013.format('roma','12.3211', '42.0582', '12.7417','41.7716'),line_2018.format('roma','12.3211', '42.0582', '12.7417','41.7716'),
                line_2013.format('vienna','16.1640','48.3350','16.5939','48.1150'),line_2018.format('vienna','16.1640','48.3350','16.5939','48.1150'),
                line_2013.format('bremen','8.4727','53.2324','8.9956','53.0078'),line_2018.format('bremen','8.4727','53.2324','8.9956','53.0078'),
                line_2013.format('dublin', '-6.4684','53.4317','-6.0260','53.2169'),line_2018.format('dublin', '-6.4684','53.4317','-6.0260','53.2169')
                    ]



    all_text = []
    all_text.extend(china_text)
    all_text.extend(europe_text)
    for cloud_cover in [20,50,100]:
        for r in all_text:
            list = run(r,cloud_cover)   #获取影像列表
            # ee = EarthExplorer(username, password)  #登陆账号
            print(list[0])
            # Download_from_Landsatexplore("landsat_8_c1", list,r)    # 下载并解压数据


    api.logout()







