from .aug_pil_method import *

def augment_list_senior_13():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 1.2),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            (Posterize, 4, 8),
            (Sharpness, 0.1, 2.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            (Detail, 0, 1),
            #(Emboss, 0, 1),
            (Contour, 0, 1),
            #(Find_edges, 0, 1),
            (Edge_enhance, 0, 1),
            # (Edge_enhance_more, 0, 1),
            (Smooth, 0, 1),
            # (Smooth_more, 0, 1),
            # (Sharpen, 0, 1),
            #(Identity, 0, 1),
            ]
    return augs

def augment_list():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 1.2),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            (Posterize, 4, 8),
            (Sharpness, 0.1, 2.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            (Detail, 0, 1),
            #(Emboss, 0, 1),
            #(Contour, 0, 1),
            #(Find_edges, 0, 1),
            (Edge_enhance, 0, 1),
            (Edge_enhance_more, 0, 1),
            (Smooth, 0, 1),
            (Smooth_more, 0, 1),
            (Sharpen, 0, 1),
            #(Identity, 0, 1),
            ]
    return augs

def augment_list1():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 3.0),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            (Posterize, 4, 8),
            (Sharpness, 0.1, 2.0), # (Sharpness, 0.5, 5.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            (Detail, 0, 1),
            (Emboss, 0, 1),
            (Contour, 0, 1),
            #(Find_edges, 0, 1),
            (Edge_enhance, 0, 1),
            (Edge_enhance_more, 0, 1),
            (Smooth, 0, 1),
            (Smooth_more, 0, 1),
            (Sharpen, 0, 1),
            #(Identity, 0, 1),
            # (gauss_noise, 0, 0.01), # 2021/12/28 加的
            ]
    return augs

def augment_list_16():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 3.0),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            # (Posterize, 4, 8),
            (Sharpness, 0.5, 5.0), # (Sharpness, 0.1, 2.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            (Detail, 0, 1),
            (Emboss, 0, 1),
            (Contour, 0, 1),
            #(Find_edges, 0, 1),
            (Edge_enhance, 0, 1),
            (Edge_enhance_more, 0, 1),
            (Smooth, 0, 1),
            (Smooth_more, 0, 1),
            (Sharpen, 0, 1),
            #(Identity, 0, 1),
            ]
    return augs

def augment_list_17():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 3.0),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            # (Posterize, 4, 8),
            (Sharpness, 0.5, 5.0), # (Sharpness, 0.1, 2.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            (Detail, 0, 1),
            (Emboss, 0, 1),
            (Contour, 0, 1),
            #(Find_edges, 0, 1),
            (Edge_enhance, 0, 1),
            (Edge_enhance_more, 0, 1),
            (Smooth, 0, 1),
            (Smooth_more, 0, 1),
            (Sharpen, 0, 1),
            #(Identity, 0, 1),
            (gauss_noise, 0, 0.01),
            ]
    return augs

def augment_list_18():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 3.0),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            (Posterize, 4, 8),
            (Sharpness, 0.5, 5.0), # (Sharpness, 0.1, 2.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            (Detail, 0, 1),
            (Emboss, 0, 1),
            (Contour, 0, 1),
            #(Find_edges, 0, 1),
            (Edge_enhance, 0, 1),
            (Edge_enhance_more, 0, 1),
            (Smooth, 0, 1),
            (Smooth_more, 0, 1),
            (Sharpen, 0, 1),
            #(Identity, 0, 1),
            (gauss_noise, 0, 0.01),
            ]
    return augs


def augment_list_best_9Aug():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 3.0),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            # (Posterize, 4, 8),
            # (Sharpness, 0.5, 5.0), # (Sharpness, 0.1, 2.0),
            (SolarizeAdd, 0, 80),
            (Blur, 0, 1),
            # (Detail, 0, 1),
            (Emboss, 0, 1),
            (Contour, 0, 1),
            #(Find_edges, 0, 1),
            # (Edge_enhance, 0, 1),
            # (Edge_enhance_more, 0, 1),
            # (Smooth, 0, 1),
            # (Smooth_more, 0, 1),
            # (Sharpen, 0, 1),
            #(Identity, 0, 1),
            ]
    return augs

def augment_list_best_1Aug():
    # Test
    augs = [(Contour, 0, 1),]
    return augs

def augment_list_new_best_2Aug():
    # Test
    augs = [(SolarizeAdd, 0, 80),
            (Contour, 0, 1),
            ]
    return augs

def augment_list_new_best_3Aug():
    # Test
    augs = [(SolarizeAdd, 0, 80),
            (Contour, 0, 1),
            (Invert, 0, 1),
            ]
    return augs    

def augment_list_new_best_4Aug():
    # Test
    augs = [(SolarizeAdd, 0, 80),
            (Contour, 0, 1),
            (Invert, 0, 1),
            (Blur, 0, 1),
            ]
    return augs    

def augment_list_best_2Aug():
    # Test
    augs = [(Blur, 0, 1),
            (Contour, 0, 1),
            ]
    return augs

def augment_list_best_3Aug():
    # Test
    augs = [(Blur, 0, 1),
            (Contour, 0, 1),
            (SolarizeAdd, 0, 80),
            ]
    return augs

def augment_list_best_4Aug():
    # Test
    augs = [(Blur, 0, 1),
            (Contour, 0, 1),
            (SolarizeAdd, 0, 80),
            (Invert, 0, 1),
            ]
    return augs

def augment_list_3Aug():
    # Test
    augs = [(Blur, 0, 1),
            (Contour, 0, 1),
            (Posterize, 4, 8),
            ]
    return augs

def augment_list_gauss_noise():
    # Test
    augs = [(gauss_noise, 0, 0.01),
            ]
    return augs

def augment_list_gauss_noise_Contour():
    # Test
    augs = [(gauss_noise, 0, 0.01),
            (Contour, 0, 1),
            ]
    return augs

def augment_list_gauss_noise_Contour_Blur():
    # Test
    augs = [(gauss_noise, 0, 0.01),
            (Contour, 0, 1),
            (Blur, 0, 1),
            ]
    return augs

def augment_list_Detail():
    # Test
    augs = [(Detail, 0, 1),
            ]
    return augs

def augment_list_space():
    augs = [(hflip, 0 , 1),
            (Rotate, -20, 20),
            (Horizontal_Shift, -0.1, 0.1),
            (Vertical_Shift, -0.1, 0.1),
            (Zoom, 0, 0.1),
            ]
    return augs

def augment_list_space_2aug():
    augs = [(hflip, 0 , 1),
            (Rotate, -20, 20),
            ]
    return augs

def augment_list_filter_2aug_space_3aug():
    augs = [(Contour, 0, 1),
            (Blur, 0, 1),
            (hflip, 0 , 1),
            (Rotate, -20, 20),
            (Zoom, 0, 0.1),
            ]
    return augs

def augment_list_M():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 1.2),
            ]
    return augs

def augment_list_senior_13_80_percent():
    # Test
    augs = [(AutoContrast, 0, 1),
            (Brightness, 0.3, 1.2),
            (Contrast, 0.3, 1.2),
            (Equalize, 0, 1),
            (Invert, 0, 1),
            (Posterize, 4, 8),
            (Sharpness, 0.1, 2.0),
            (SolarizeAdd, 0, 80),
            # (Blur, 0, 1),
            (Detail, 0, 1),
            #(Emboss, 0, 1),
            # (Contour, 0, 1),
            #(Find_edges, 0, 1),
            (Edge_enhance, 0, 1),
            # (Edge_enhance_more, 0, 1),
            (Smooth, 0, 1),
            # (Smooth_more, 0, 1),
            # (Sharpen, 0, 1),
            #(Identity, 0, 1),
            ] + [
            (Blur, 0, 1),
            ] * 22 + [
            (Contour, 0, 1),
            ] * 22
    return augs

if __name__ == "__main__":
    print(augment_list_senior_13_80_percent())
    