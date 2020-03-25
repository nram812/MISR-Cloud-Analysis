"""here is a script for reading the satellite imagery from MISR"""
def open_misr_imagery(directory,block_0,block_1):

    #block_0 is the first block of the imagey.
    #Often the first block has invalid data, to view only the valid blocks, use the metadata commands shown below,
    #for the entire image, select block_0=0 and block_1 as 180.
    #Use the metadata to retrieve the blocks.


    #NOTE: opening a large amount of blocks results to a slow processing time.

    #importing the generic modules
    import MisrToolkit as mtk

    mfile = (directory)
    m = mtk.MtkFile(mfile)
    path = m.path
    #here is the red Equivalent reflectance field
    #to find the other fields use m.grid_list or m.grid('RedBand').field_list
    #full documentation is available on the MISR toolkit.
    fld = m.grid('RedBand').field('Red Equivalent Reflectance')


    #here is some example code for retrieving the block metadata fields.
    #m.block_metadata_list
    #m.block_metadata_field_list('PerBlockMetadataCommon')
    #block_times = m.block_metadata_field_read('PerBlockMetadataTime', 'BlockCenterTime')
    #block_numbers = m.block_metadata_field_read('PerBlockMetadataCommon', 'Block_number')
    #valid_data = m.block_metadata_field_read('PerBlockMetadataCommon', 'Data_flag')
    #Ocean_Blocks = np.array(m.block_metadata_field_read('PerBlockMetadataCommon', 'Ocean_flag'))

    mtk_region = mtk.MtkRegion(path, block_0, block_1)  # using mtk_region to stich the blocks
    imagery = fld.read(mtk_region).data()
    return imagery

import matplotlib.pyplot as plt
directory='write your own directory'
imagery=open_misr_imagery(directory,10,30)
fig,ax=plt.subplots(10,10,dpi=100)

#cmap, changes the colour scale

#min=0,vmax=1,cmap='gray')
ax.imshow(imagery,v



