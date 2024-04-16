'''
原作者地址：https://github.com/anothermartz/Easy-Wav2Lip
眠NEON-优化版，B站地址：https://space.bilibili.com/386198029 \Wechat:dzemcn
'''

import os, sys
import gradio as gr
import configparser
from pathlib import Path
import shutil
import subprocess

def read_config(config_path='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_path)
    settings = {
        'quality': config.get('OPTIONS', 'quality', fallback='Improved'),
        'output_height': config.get('OPTIONS', 'output_height', fallback='full resolution'),
        'wav2lip_version': config.get('OPTIONS', 'wav2lip_version', fallback='Wav2Lip'),
        'use_previous_tracking_data': config.getboolean('OPTIONS', 'use_previous_tracking_data', fallback=True),
        'nosmooth': config.getboolean('OPTIONS', 'nosmooth', fallback=True),
        'u': config.getint('PADDING', 'u', fallback=0),
        'd': config.getint('PADDING', 'd', fallback=10),
        'l': config.getint('PADDING', 'l', fallback=0),
        'r': config.getint('PADDING', 'r', fallback=0),
        'size': config.getfloat('MASK', 'size', fallback=2.5),
        'feathering': config.getint('MASK', 'feathering', fallback=2),
        'mouth_tracking': config.getboolean('MASK', 'mouth_tracking', fallback=False),
        'debug_mask': config.getboolean('MASK', 'debug_mask', fallback=False),
        'batch_process': config.getboolean('OTHER', 'batch_process', fallback=False),
    }
    return settings
    
def update_config_file(config_values):
    quality, output_height, wav2lip_version, use_previous_tracking_data, nosmooth, u, d, l, r, size, feathering, mouth_tracking, debug_mask, batch_process, source_image, driven_audio = config_values

    config = configparser.ConfigParser()
    config.read('config.ini')

    config.set('OPTIONS', 'video_file', str(source_image))
    config.set('OPTIONS', 'vocal_file', str(driven_audio))
    config.set('OPTIONS', 'quality', str(quality))
    config.set('OPTIONS', 'output_height', str(output_height))
    config.set('OPTIONS', 'wav2lip_version', str(wav2lip_version))
    config.set('OPTIONS', 'use_previous_tracking_data', str(use_previous_tracking_data))
    config.set('OPTIONS', 'nosmooth', str(nosmooth))
    config.set('PADDING', 'U', str(u))
    config.set('PADDING', 'D', str(d))
    config.set('PADDING', 'L', str(l))
    config.set('PADDING', 'R', str(r))
    config.set('MASK', 'size', str(size))
    config.set('MASK', 'feathering', str(feathering))
    config.set('MASK', 'mouth_tracking', str(mouth_tracking))
    config.set('MASK', 'debug_mask', str(debug_mask))
    config.set('OTHER', 'batch_process', str(batch_process))
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

def copy_to_folder(uploaded_file, target_folder):
    if isinstance(uploaded_file, gr.Files):
        file_path = Path(uploaded_file.name).resolve()
    else:
        file_path = Path(uploaded_file).resolve()

    target_path = Path(target_folder) / file_path.name
    shutil.copy(str(file_path), str(target_path))
    return str(target_path)

def run_wav2lip():
    python_executable = sys.executable
    subprocess.run([python_executable, 'run.py'])
    video_files = list(Path('out').glob('*.mp4'))
    if not video_files:  
        return None, "未找到文件❗"

    latest_video_file = max(video_files, key=lambda x: x.stat().st_mtime)
    return str(latest_video_file), "成功了！😭"

def execute_pipeline(source_media, driven_audio, quality, output_height, wav2lip_version, 
                     use_previous_tracking_data, nosmooth, u, d, l, r, size, feathering, 
                     mouth_tracking, debug_mask, batch_process):
    if os.path.exists('in'):
        shutil.rmtree('in')
    os.makedirs('in', exist_ok=True)

    source_media_path = copy_to_folder(source_media, 'in')
    driven_audio_path = copy_to_folder(driven_audio, 'in')

    config_values = (quality, output_height, wav2lip_version, use_previous_tracking_data, nosmooth, 
                     u, d, l, r, size, feathering, mouth_tracking, debug_mask, batch_process,
                     source_media_path, driven_audio_path)

    update_config_file(config_values)
    video_path, message = run_wav2lip()
    return video_path, message

    

def easywav2lip_demo( config_path='config.ini'):
    settings = read_config(config_path)
    with gr.Blocks(analytics_enabled=False) as easywav2lip_interface:
        gr.Markdown(" <h2>Easy_Wav2Lip交互后台</span> </h2> <p style='font-size:18px;color: black;'>作者：眠</p></div>") 
        
        with gr.Row(equal_height=False):
            with gr.Tabs(elem_id="source_media"):
                with gr.TabItem('上传原素材'):
                    with gr.Row():
                        source_media = gr.File(label="支持图片、视频格式", file_types=['image','video'] ,type="filepath", elem_id="source_media")

            with gr.Tabs(elem_id="driven_audio"):
                with gr.TabItem('上传音频'):
                    with gr.Column(variant='panel'):
                            driven_audio = gr.Audio(label="支持mp3、wav格式", sources="upload", type="filepath")
        with gr.Row(equal_height=False):                    
            with gr.Tabs(elem_id="easywav2lip_checkbox"):
                with gr.TabItem('设置'):
                    gr.Markdown("所有视频帧中必须都有一张脸，否则 Wav2Lip 将报错。第一次使用，请测试音频较短的文件，再进行长视频制作")
                    with gr.Row(variant='panel'): 
                        with gr.Column(variant='panel'):
                                # 使用从config.ini文件中读取的默认值来初始化Gradio组件
                            quality = gr.Radio(
                                ['Fast', 'Improved', 'Enhanced', 'Experimental'], 
                                value=settings['quality'], 
                                label='视频质量选项', 
                                info="与视频生成速度和清晰度有关"
                            )
                            output_height = gr.Radio(
                                ['full resolution','half resolution'],
                                value=settings['output_height'], label='分辨率选项',
                                info="全分辨率和半分辨率"
                            )
                            wav2lip_version = gr.Radio(
                                ['Wav2Lip','Wav2Lip_GAN'],
                                value=settings['wav2lip_version'], label='Wav2Lip版本选项',
                                info="Wav2Lip口型同步更好。若出现牙齿缺失，可尝试Wav2Lip_GAN")
                            use_previous_tracking_data = gr.Radio(
                                ['True', 'False'],
                                value='True' if settings['use_previous_tracking_data'] else 'False', 
                                label='启用追踪旧数据'
                            )
                            nosmooth = gr.Radio(
                                ['True', 'False'],
                                value='True' if settings['nosmooth'] else 'False', 
                                label='启用脸部平滑', 
                                info="适用于快速移动的人脸，如果脸部角度过大可能会导致动画抽搐"
                            )
                            batch_process = gr.Radio(
                                ['False'],
                                value='True' if settings['batch_process'] else 'False', 
                                label='批量处理多个视频',
                                info="目前webui版本暂不支持批量处理，您可运行源代码"
                            )
                                
                        with gr.Column(variant='panel'): 
                            with gr.Column(variant='panel'):        
                                Padding_u = gr.Slider(label="嘴部mask上边缘", step=1, minimum=-100, maximum=100,value=int(settings['u']))
                                Padding_d = gr.Slider(label="嘴部mask下边缘", step=1, minimum=-100, maximum=100,value=int(settings['d']))
                                Padding_l = gr.Slider(label="嘴部mask左边缘", step=1, minimum=-100, maximum=100,value=int(settings['l']))
                                Padding_r = gr.Slider(label="嘴部mask右边缘", step=1, minimum=-100, maximum=100,value=int(settings['r']))
                                mask_size = gr.Slider(label="mask尺寸", step=0.1, minimum=-10, maximum=10,value=float(settings['size']), info="减小脸部周围的边框")
                                mask_feathering = gr.Slider(label="mask羽化", step=1, minimum=-100, maximum=100,value=float(settings['feathering']), info="减轻脸部周围边框的清晰度")
                                mask_mouth_tracking = gr.Radio(
                                    ['True', 'False'],
                                    value='True' if settings['mouth_tracking'] else 'False', 
                                    label='启用mask嘴部跟踪'
                                )
                                mask_debug_mask = gr.Radio(
                                    ['True', 'False'],
                                    value='True' if settings['debug_mask'] else 'False', 
                                    label='启用mask调试'
                                )
                                
                        with gr.Column(variant='panel'): 
                            with gr.Column(variant='panel'):           
                                gen_video = gr.Video(label="视频生成", format="mp4",height=500,width=450)
                                output_message = gr.Textbox(label='视频制作状态')
                                submit_btn = gr.Button('🚀一键三连', elem_id="easywav2lip_generate", variant='primary')
                                

                submit_btn.click(
                    fn=execute_pipeline,
                    inputs=[source_media, driven_audio, quality, output_height, wav2lip_version, use_previous_tracking_data, nosmooth, Padding_u, Padding_d, Padding_l, Padding_r, mask_size, mask_feathering, mask_mouth_tracking, mask_debug_mask, batch_process],
                    outputs=[gen_video, output_message]
                )

    return easywav2lip_interface
 

if __name__ == "__main__":

    demo = easywav2lip_demo()
    demo.queue()
    demo.launch()