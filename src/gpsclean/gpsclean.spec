# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(['gpsclean.py'],
             pathex=['C:\\Users\\Davide\\Documents\\unibz\\2_Master\\Tesi\\GPSClean_github\\gpsclean\\src\\gpsclean'],
             binaries=[],
             datas=[('data/model_42t_traces.h5', 'data')],
             hiddenimports=['tensorflow.python.keras.engine.base_layer_v1'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='gpsclean',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )