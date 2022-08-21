from osgeo import ogr

shpFile = 'C:\\Users\owl\Desktop\planet_116.19_39.753_66eefaf9-shp (1)\shape\landuse.shp'  # 裁剪矩形

# # # 注册所有的驱动
ogr.RegisterAll()


def check_shp():
    # 打开数据
    ds = ogr.Open(shpFile, 0)
    if ds is None:
        print("打开文件【%s】失败！", shpFile)
        return
    print("打开文件【%s】成功！", shpFile)
    # 获取该数据源中的图层个数，一般shp数据图层只有一个，如果是mdb、dxf等图层就会有多个
    m_layer_count = ds.GetLayerCount()
    m_layer = ds.GetLayerByIndex(0)
    if m_layer is None:
        print("获取第%d个图层失败！\n", 0)
        return
    # 对图层进行初始化，如果对图层进行了过滤操作，执行这句后，之前的过滤全部清空
    m_layer.ResetReading()
    count = 0
    m_feature = m_layer.GetNextFeature()
    while m_feature is not None:
        o_geometry = m_feature.GetGeometryRef()
        if not ogr.Geometry.IsValid(o_geometry):
            print(m_feature.GetFID())
            count = count + 1

        m_feature = m_layer.GetNextFeature()
    print("无效多边形共" + str(count) + "个")
check_shp()