import moviepy
from moviepy.video.VideoClip import ImageClip
from moviepy.editor import VideoFileClip, concatenate_videoclips, ipython_display
from PIL import Image, ImageChops
import numpy as np
from os.path import join as opj
from collections import OrderedDict
import matplotlib.pyplot as plt


from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def supplement_ims(ims, combine_func = ImageChops.soft_light, combine_func_args={}):
    n_ims = len(ims)
    new_ims = [ims[0]]
    for i in range(0, n_ims-1):
        new_ims.append(combine_func(ims[i], ims[i+1], **combine_func_args))
        new_ims.append(ims[i+1])
    return new_ims
        
#get sub-images
def get_sub_images(left, bottom, sub_image_size, im_files, right=None, size_out=None):
    box = [left, 1.-bottom-sub_image_size, left+sub_image_size, 1-bottom]
    orig_size=Image.open(im_files[0]).size
    box = [int(box[0]*orig_size[0]+0.5), int(box[1]*orig_size[1]+0.5), int(box[2]*orig_size[0]+0.5), int(box[3]*orig_size[1]+0.5)]
    ims_orig = [(Image.open(im_file).convert('L').convert('P')
).crop(tuple(box)) for im_file in im_files]
    if size_out is not None:
        ims_orig = [im.resize(size_out,resample=Image.LANCZOS) for im in ims_orig]
    return ims_orig



#few seconds of pear
def get_switch_fam(rng, probs, current_fam):
    fam_names = list(probs.keys())
    new_fam_ind = rng.choice(np.arange(len(fam_names)),
                   p=np.array(probs[current_fam]))
    return fam_names[new_fam_ind]


def get_decision(rng, prob):
    return rng.random()<prob

def seconds_to_milliseconds(t):
    return t*1000

def update_index(current_index, nims_current_fam, 
                 reverse):
    if reverse:
        new_index = current_index-1
    else:
        new_index = current_index+1
    return new_index



def mix_frames(image_families, 
               switch_fam_probs,
               dts,
               film_length=5., 
               end_at_frame=None,
              seed=1234,
              start_im_fam=None,
              start_reverse=False,
              reverse_at_end_of_family=["pears","text"],
              reverse_prob=0.,
              fam_start_inds = None,
              reverse_prob_if_reverse=None,
              advance_after_fam_switch=False,
              verbose=True):
    
    """
    Parameters
    -----------
    film_length (float): length of film in seconds
    dt: minimum frame length in seconds
    """
    rng = np.random.default_rng(seed=seed)
    if start_im_fam == None:
        start_im_fam = list(image_families.keys())[0]
    if fam_start_inds == None:
        fam_start_inds = OrderedDict([(key,0) for key in image_families.keys()])
    print("start_im_fam",start_im_fam)
    print("fam_start_inds",fam_start_inds)

    #we collect frames (i.e. images) and 
    #durations with which to construct the gif
    #we're also going to set a total length 
    #at which to quit 
    frames = []
    durations = []
    frame_ids = []
    fams=[]
    length_so_far = 0.
    
    #since we switch between image families
    #and want to return to the same place, 
    #record the current index where we left off
    current_im_fam = start_im_fam #starting image category
    
    reverse_fam = OrderedDict([
        (f, start_reverse) for f in list(image_families.keys())]
    )
    
    #reverse=start_reverse
    fam_current_inds = fam_start_inds.copy()
    #for c in iamge_families:
    #    fam_current_inds[c] = 0
    
    
    end_at_frame_count = 0
    if end_at_frame is not None:
        print("ending when %s,%d count is %d"%end_at_frame)
    while True:
        #get the images for the current family
        current_fam_ims = image_families[current_im_fam]
        #get the current index 
        current_ind = fam_current_inds[current_im_fam]

        #check finishing condition(s)
        if film_length is not None:
            if length_so_far>film_length:
                break
        else:
            pass
        
        #get the current image
        im = current_fam_ims[current_ind]
        nims_current_fam = len(current_fam_ims)

        try:
            duration = np.random.choice(
                dts[current_im_fam][0], 
                p=dts[current_im_fam][1]
            )
        except TypeError:
            duration = dts
        frame_id = (current_im_fam, current_ind)
        if verbose:
            print("adding frame:",frame_id)
            frames.append(im)
            durations.append(duration)
            frame_ids.append(frame_id)
            fams.append(current_im_fam)

        if end_at_frame is not None:
            if (current_im_fam, current_ind) == (end_at_frame[0],end_at_frame[1]):
                end_at_frame_count += 1
                print("reached end frame, count is %d"%end_at_frame_count)
                if end_at_frame_count == end_at_frame[2]:
                    break
        
        #decide whether to switch family
        new_im_fam = get_switch_fam(rng, switch_fam_probs,
                                    current_im_fam)
        
        if new_im_fam != current_im_fam:
            if verbose:
                print("switching family to %s"%new_im_fam)
            #and note that we've changed frames
            changed_frames = True
            if advance_after_fam_switch:
                new_ind = fam_current_inds[new_im_fam]+1
                if new_ind>len(image_families[new_im_fam])-1:
                    new_ind=0
                fam_current_inds[new_im_fam]=new_ind
            if verbose:
                print("next frame ind should be %d"%(fam_current_inds[new_im_fam]))
        else:
            #otherwise decide whether to progress frame
            new_im_fam = current_im_fam
            if verbose:
                print("sticking with family %s"%new_im_fam)
            change_direction = get_decision(rng, reverse_prob)
            if change_direction:
                reverse_fam[current_im_fam] = not reverse_fam[current_im_fam]
            #if current_im_fam in reverse_at_end_of_family:
            #    if ((current_ind == nims_current_fam-1)
            #        or (current_ind==0)):
            #        reverse_fam[current_im_fam] = not reverse_fam[current_im_fam]
            if verbose:
                print("reverse:", reverse_fam[current_im_fam])
            #if we do, need to update current frame index 
            #fam_current_inds[current_im_fam]
            #new_ind = (current_ind + 1)%nims_current_fam
            new_ind = update_index(current_ind, nims_current_fam, reverse_fam[current_im_fam])
            if nims_current_fam==1:
                new_ind=0
            if new_ind>nims_current_fam-1:
                new_ind-=2
                reverse_fam[current_im_fam]=True
            elif new_ind<0:
                new_ind=1
                reverse_fam[current_im_fam]=False
            fam_current_inds[current_im_fam] = new_ind
            if verbose:
                print("progressing frame to index %d"%fam_current_inds[current_im_fam])
            

            
        current_im_fam = new_im_fam
            
        length_so_far = np.sum(np.array(durations))
        #check finishing condition(s)

        print("******************")
        
    return frames, np.array(durations)*1000, frame_ids, fams
    
def lighten_im(im, factor):
    im_array = np.asarray(im)
    im_out_array = 1-factor*(1-im_array)
    return Image.fromarray(np.uint8(im_out_array)).convert('P')

def scroll_im_frames(ims, n_frames, background_ims=None,
             combine_func = ImageChops.lighter, scale_ims=None,
                    scroll_duration_sec=0.2):
    frames = []
    if not isinstance(ims, list):
        ims=[ims]
    size = ims[0].size
    if len(ims)==1:
        ims *= n_frames
    for i_frame in range(n_frames):
        out_array = np.zeros(size)
        im_array=np.asarray(ims[i_frame]).T
        top = int( (1-float(i_frame)/n_frames)*size[1])
        out_array[:,:top] = im_array[:,size[1]-top:]
        out_array[:,top:] = im_array[:,::-1][:, :size[1]-top]
        #out_array[:,:bottom] = im_array[:,::-1][:,size[1]-bottom:]
        #out_array[:,:bottom] = im_array[:,:bottom]
        #out_array[:,bottom:] = im_array[:,::-1][:,:size[1]-bottom]
        if scale_ims is not None:
            out_array*=scale_ims
        out_im=Image.fromarray(np.uint8(out_array).T).convert("P")
        frames.append(out_im)
    if background_ims is not None:
        if not isinstance(background_ims, list):
            background_ims=[background_ims]
        if len(background_ims)==1:
            background_ims *= n_frames
        assert len(background_ims)==n_frames
        try:
            frames = [combine_func(b,f).convert("P") for (b,f) in zip(
                background_ims, frames)]
        except ValueError:
            frames = [combine_func(b,g_to_rgb(f)).convert("RGB") for (b,f) in zip(
                background_ims, frames)]
    return frames, [scroll_duration_sec*1000]*len(frames)

def scroll_im_frames_old(ims, n_frames, background_ims=None,
             combine_func = ImageChops.lighter, scale_ims=None):
    frames = []
    if not isinstance(ims, list):
        ims=[ims]
    size = ims[0].size
    if len(ims)==1:
        ims *= n_frames
    for i_frame in range(n_frames):
        out_array = np.zeros(size)
        im_array=np.asarray(ims[i_frame]).T
        bottom = int((1-float(i_frame)/n_frames)*size[1])
        out_array[:,bottom:] = im_array[:,:size[1]-bottom]
        out_array[:,:bottom] = im_array[:,::-1][:,size[1]-bottom:]
        if scale_ims is not None:
            out_array*=scale_ims
        out_im=Image.fromarray(np.uint8(out_array).T).convert("P")
        frames.append(out_im)
    if background_ims is not None:
        if not isinstance(background_ims, list):
            background_ims=[background_ims]
        if len(background_ims)==1:
            background_ims *= n_frames
        assert len(background_ims)==n_frames
        frames = [combine_func(b,f).convert("P") for (b,f) in zip(
            background_ims, frames)]
    return frames

def add_scroll_to_text(frames_in, durations_in, fams_in, lighten_fac=0.2, combine_func=ImageChops.soft_light,
                      ims_dict=None, scroll_frame_duration_sec=0.2):
    frames_out=[]
    durations_out=[]
    fams_out=[]
    for i,f in enumerate(frames_in):
        if fams_in[i]=="text":
            num_frames = int(durations_in[i]/(1000*scroll_frame_duration_sec))
            #scroll_frames = scroll_im_frames(lighten_im(frames_in[i-1], lighten_fac), 
            #                                 num_frames, background_ims=f, combine_func=combine_func)
            scroll_fam = fams_in[i-1]
            print("scroll_fam:", scroll_fam)
            print("num_frames:",num_frames)
            if ims_dict is not None:
                ims = ims_dict[scroll_fam]
                ims = (ims  * (num_frames // len(ims) + 1))[:num_frames]
            else:
                ims = [frames_in[i-1]]*num_frames
            scroll_frames,_ = scroll_im_frames([lighten_im(im, lighten_fac) for im in ims], 
                                             num_frames, background_ims=f, combine_func=combine_func)
            
            frames_out+=scroll_frames[:num_frames]
            durations_out+=[scroll_frame_duration_sec*1000]*num_frames
            fams_out+="text"
        else:
            frames_out.append(frames_in[i])
            durations_out.append(durations_in[i])
            fams_out.append(fams_in[i])
    return frames_out, durations_out, fams_out


def get_mono_colormap(rgb):
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(rgb[0]/256, 1, N)
    vals[:, 1] = np.linspace(rgb[1]/256, 1, N)
    vals[:, 2] = np.linspace(rgb[2]/256, 1, N)
    return ListedColormap(vals)



def g_to_mono(im, rgb=None, colormap=None, gamma=1.):
    im_array = np.array(im)
    if "A" in im.getbands():
        print("image has alpha channel")
        im_array = im_array[:,:,0]
        alpha = im.getchannel("A")
    else:
        print("image doesn't have alpha channel")
        alpha = None
        
    if colormap is not None:
        color_im_array = colormap(im_array)
        return Image.fromarray((color_im_array[:, :, :3] * 255).astype(np.uint8))
    else:
        positions = [0.,1.]
        colors = [(1,1,1),(rgb[0]/255, rgb[1]/255, rgb[2]/255)]
        print(colors)
        cmap = LinearSegmentedColormap.from_list('custom_colormap', colors[::-1])
        cmap.set_gamma(gamma)
                                                
        color_im_array = cmap(im_array)
        im_return = Image.fromarray((color_im_array[:, :, :3] * 255).astype(np.uint8))
        print(im_return.size)
        if alpha is not None:
            print(alpha.size)
            im_return.putalpha(alpha)
        return im_return

def color_mix_frames(frames_in, durations_in, cmap1, cmap2):
    frames_out = []
    durations_out = []
    n_frames = len(frames_in)
    for i in range(n_frames-1):
        im_1 = g_to_mono(frames_in[i], colormap=cmap1)
        im_2 = g_to_mono(frames_in[i+1], colormap=cmap2)
        frames_out.append(ImageChops.soft_light(im_1,im_2))
        durations_out.append(durations_in[i])
    return frames_out, durations_out

def g_to_rgb(img_g):
    img_rgb = Image.new("RGB", img_g.size)
    img_rgb.paste(img_g)
    return img_rgb
