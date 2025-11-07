"""Event configuration for pick-and-place task."""

from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg
from isaaclab.managers import SceneEntityCfg

# Import reset functions from MDP module
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place import mdp
from source.ur10e_sim2real.ur10e_sim2real.tasks.manager_based.pick_place.mdp.rewards import REACH_POSITION_SUCCESS_THRESHOLD, REACH_ROTATION_SUCCESS_THRESHOLD

OBJECT_POSE_RANGE = {
    "x": (-0.25, 0.25),      # 0.0 ± 0.25 = [-0.25, 0.25]
    "y": (0.25, -0.25),      # -0.60 + 0.25 = -0.35 (outside 300 mm pinching hazard zone) / -0.60 - 0.25 = -0.85 (still on table)
    "z": (0.0, 0.0), 
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (-3.14, 3.14),
}

OBJECT_VELOCITY_RANGE = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}

@configclass
class ReachStageSuccessMetricsEventCfg:
    """Event configuration for resets and randomization."""
    initialize_curriculum = EventTermCfg(
        func=mdp.initialize_curriculum,
        mode="startup"
    )
    update_curriculum = EventTermCfg(
        func=mdp.update_curriculum,
        mode="interval",
        interval_range_s=(0, 0),  # check every step
    )
    update_success_metrics = EventTermCfg(
        func=mdp.update_success_metrics,
        mode="interval",
        interval_range_s=(0, 0), # Check every step
        params={
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "position_threshold": REACH_POSITION_SUCCESS_THRESHOLD,
            "rotation_threshold": REACH_ROTATION_SUCCESS_THRESHOLD,
        },
    )
    reset_episodic_success_metrics = EventTermCfg(
        func=mdp.reset_success_counters,
        mode="reset",
    )
    
@configclass
class ReachStageEventCfg(ReachStageSuccessMetricsEventCfg):
    # Reset object and target poses
    reset_object = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": OBJECT_POSE_RANGE,
            "velocity_range": OBJECT_VELOCITY_RANGE,
        },
    )

    # Reset robot to home position
    reset_articulation_to_default = EventTermCfg(
        func=mdp.reset_articulation_to_default,
        mode="reset",
    )

@configclass
class ReachStageDeterministicEventCfg(ReachStageSuccessMetricsEventCfg):
    # Reset object and target poses
    reset_object = EventTermCfg(
        func=mdp.reset_root_state_predefined,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "poses": [
                # 100 predefined poses uniformly sampled within OBJECT_POSE_RANGE (seed=999)
                {'x': 0.15171402003984397, 'y': 0.013761147841322374, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.0536227246198828},
                {'x': -0.036309526279331594, 'y': 0.02719290621532633, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -2.311728583821854},
                {'x': -0.0786147741836476, 'y': -0.1492201945922333, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -0.5955910245382757},
                {'x': 0.13021773516965546, 'y': -0.013120812119237879, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -0.550393089400279},
                {'x': -0.1091805187201888, 'y': -0.0548611097051539, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.29851527105089504},
                {'x': -0.08389846519079391, 'y': 0.23663440676891073, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.906394575522467},
                {'x': 0.1704100053678393, 'y': -0.24812339840404007, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 2.9679384452002795},
                {'x': 0.16641152096796524, 'y': -0.2170911970626992, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 3.001067306686184},
                {'x': 0.159495104470265, 'y': -0.15622937795261593, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -0.10907631129894672},
                {'x': 0.11529633792282584, 'y': 0.14759117822530565, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 2.5089176119721435},
                {'x': 0.2075778092444952, 'y': -0.09377549247540878, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 1.4164258730709225},
                {'x': 0.04082961420665032, 'y': 0.22728434819210702, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 2.2505216779724178},
                {'x': -0.02547689652232793, 'y': 0.16182518853591854, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -0.4771103400171742},
                {'x': 0.1977439320771795, 'y': -0.02781367218713887, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -2.9958255527392876},
                {'x': 0.16883468657480988, 'y': -0.09023354016300222, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.9789894757076332},
                {'x': 0.2348434160353814, 'y': 0.0982228040203289, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.5044251166787825},
                {'x': 0.07633744181795499, 'y': 0.061486811320729196, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -2.7420990250486392},
                {'x': 0.18356788056173273, 'y': 0.025737502811984225, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -0.07389064815353118},
                {'x': 0.04543616493025765, 'y': 0.18543647946844288, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -2.469063620596109},
                {'x': 0.0635796992723579, 'y': -0.06282744213077918, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 1.0413167849777403},
                {'x': 0.2267504147522068, 'y': -0.15922762709777727, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -2.5600116001588344},
                {'x': 0.08605408524326075, 'y': 0.206734233165691, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -2.788759423910441},
                {'x': -0.21961432664118435, 'y': 0.13991685302038598, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.985221610065105},
                {'x': 0.2170631986640899, 'y': 0.021561975461768745, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -0.015838705148936465},
                {'x': -0.16680318791757176, 'y': 0.229620914282535, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 2.7258101527586454},
                {'x': -0.05755519284881527, 'y': -0.07426387475044993, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.8935583649913953},
                {'x': 0.2376100909563051, 'y': -0.21328037525205001, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 3.016031097539657},
                {'x': -0.1447673228237089, 'y': 0.2173505867207337, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.13680444170787315},
                {'x': -0.050591090112889814, 'y': 0.13628373228111101, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.936248335358809},
                {'x': 0.1423759969645229, 'y': 0.0878617821265183, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 2.010839216207458},
                {'x': -0.019419035706925547, 'y': 0.07899438697435035, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -0.635004271355615},
                {'x': 0.1966732933465022, 'y': -0.11526940873307318, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -0.07595629859540758},
                {'x': 0.23894765984490907, 'y': 0.007958516325136666, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -0.7316933476220028},
                {'x': 0.17793944572148535, 'y': -0.22070046606981875, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 1.536480685694074},
                {'x': -0.19791699124046652, 'y': 0.2296138748478468, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.8546500894055807},
                {'x': 0.137391525626476, 'y': -0.10667096157098094, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -2.6558221870322045},
                {'x': 0.020570293634203263, 'y': -0.24736938476290216, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.4670727692463514},
                {'x': -0.1694253441965563, 'y': 0.02970374215364796, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.08661460034388883},
                {'x': -0.12077861046133104, 'y': 0.1389354655592865, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.7304675320885072},
                {'x': -0.17723284148732732, 'y': -0.1297598349955379, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 2.0076019483529803},
                {'x': -0.09180207678041596, 'y': 0.1696468683734435, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.5849594011915529},
                {'x': 0.07575851385179849, 'y': 0.1965177924780579, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.40907845902974005},
                {'x': 0.11545123286339887, 'y': 0.15939062661412007, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 3.0804698691416874},
                {'x': -0.054348868901000824, 'y': 0.19352734919704806, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.6885023034086896},
                {'x': -0.20776324774645477, 'y': -0.039662647885259084, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 1.1530313071348892},
                {'x': -0.22154901754043665, 'y': 0.11969490598831872, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -2.37839141283154},
                {'x': 0.2160197046705541, 'y': 0.10536082963195537, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -0.9583190008666382},
                {'x': 0.2204651124512697, 'y': -0.2091013274740136, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 1.3095245708236347},
                {'x': -0.22800310757090186, 'y': 0.010726446365155573, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -2.3147211132163075},
                {'x': -0.029318359275620043, 'y': 0.08034291401095911, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -2.3746732276131883},
                {'x': 0.20986723443485183, 'y': 0.2117005474016337, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.1157108821261865},
                {'x': 0.19405255896546658, 'y': 0.19645340924146348, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -0.8584990715705466},
                {'x': -0.0003663986237403827, 'y': 0.1282516406328239, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -2.6365993351618884},
                {'x': 0.04073731180867879, 'y': 0.061759013467497736, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -2.7831819720155826},
                {'x': 0.10740843716606424, 'y': 0.2050241815932578, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -2.4706021112577736},
                {'x': -0.08497679056688656, 'y': -0.15948506730441808, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 1.3987613531016647},
                {'x': -0.05612560952995621, 'y': 0.012239611763608549, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.2375667689247372},
                {'x': -0.04531182506042947, 'y': -0.20143777237254984, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -3.0206595230448454},
                {'x': 0.11307363126361508, 'y': 0.20943384784227492, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 1.4738068868468839},
                {'x': 0.05292625012114294, 'y': 0.21377661519616453, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -3.0824772337605544},
                {'x': -0.2044126091784682, 'y': 0.053506789168844715, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 2.158695474066718},
                {'x': 0.21062778904119883, 'y': -0.005166702484982577, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -0.8915222559975461},
                {'x': 0.24900572952571648, 'y': 0.08260371037548347, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 2.0654377858546855},
                {'x': 0.14010060504949862, 'y': -0.0783011262372183, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.5652538860030971},
                {'x': 0.2420726781952564, 'y': -0.09697573332209186, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -0.8054474554980868},
                {'x': 0.0013983741129681704, 'y': -0.06137826041693295, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -0.979797254457872},
                {'x': -0.24424836458921823, 'y': -0.12931498243620326, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 1.4099992005184638},
                {'x': -0.1550988897261139, 'y': -0.004144505430711931, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.3105370362173248},
                {'x': -0.013183592161383761, 'y': 0.029037175029109796, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -3.0999493829600993},
                {'x': -0.24924574140201572, 'y': -0.07783711458397502, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.5523382987215456},
                {'x': -0.09330840254030787, 'y': -0.15890578198570054, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.7897896131934855},
                {'x': 0.07835604029003351, 'y': -0.22689622675544274, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 2.039184212466481},
                {'x': 0.022185535874991114, 'y': -2.2677407894466928e-05, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0646981899516072},
                {'x': -0.07622226687510797, 'y': 0.07287436989811402, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.3197558733876521},
                {'x': -0.11815840516168363, 'y': 0.19487127865066217, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.4113399112482559},
                {'x': 0.13619457979928323, 'y': -0.18565760101033119, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 2.1163494097081115},
                {'x': -0.1909081330097851, 'y': 0.1288513787364846, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -0.4234143207462111},
                {'x': -0.035363789230286435, 'y': -0.11846673996038909, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 2.236610126618275},
                {'x': -0.006666001048623804, 'y': -0.006168818434965806, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.0323177238264551},
                {'x': 0.15130518335436577, 'y': 0.2483147436955626, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.740558605130737},
                {'x': -0.08848268476456517, 'y': 0.2134971019395716, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 1.1600738263259032},
                {'x': -0.12062644380208631, 'y': -0.036069103039352735, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.9562651783613414},
                {'x': 0.10766423683239673, 'y': -0.10689892886808305, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.663429475612431},
                {'x': 0.073439101844257, 'y': -0.08791831457748633, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 2.4691965061483674},
                {'x': 0.03890814139019061, 'y': -0.13646848743727763, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -2.6133708930060475},
                {'x': -0.22803071685947135, 'y': -0.20944725574181394, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.8683409876922188},
                {'x': 0.014058324869224037, 'y': 0.11561956509259907, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.6660653012909258},
                {'x': -0.10704374455257187, 'y': -0.17955266809504833, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.1737132902889296},
                {'x': -0.22389337119852998, 'y': -0.16537586786823666, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.3568241962971437},
                {'x': -0.12089970912302878, 'y': 0.14147005268990892, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 3.073059737301233},
                {'x': -0.1989941976790227, 'y': 0.11847822604952168, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 2.259598239040921},
                {'x': 0.024526796245399884, 'y': -0.17478979796383703, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.06543870586181273},
                {'x': -0.04486829496200129, 'y': 0.12398130724065454, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -2.878537980364181},
                {'x': 0.0893185892324298, 'y': -0.24674319695697217, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -2.4330772744638676},
                {'x': -0.17299693262304044, 'y': 0.18823311558442302, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.8328823054816943},
                {'x': -0.14498247324803987, 'y': 0.23440163009715204, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -0.1799640425748296},
                {'x': 0.013218243826949583, 'y': -0.18149741272398012, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 3.1293723435893397},
                {'x': 0.06977181117683234, 'y': -0.1323667018208074, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.7778991934355302},
                {'x': 0.24414076145469682, 'y': 0.009990650833823278, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.805570841516611},
                {'x': -0.19249622648634235, 'y': 0.05504168581818375, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': -1.026019685404648},
            ],
        },
    )

    # Reset robot to home position
    reset_articulation_to_default = EventTermCfg(
        func=mdp.reset_articulation_to_default,
        mode="reset",
    )
    
@configclass
class ReachStageRandomizeObjectOnSuccessEventCfg(ReachStageEventCfg):
    reset_object_on_success = EventTermCfg(
        func=mdp.reset_object_on_success,
        mode="interval",
        interval_range_s=(0, 0), # Check every step
        params={
            "object_cfg": SceneEntityCfg("object"),
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "position_threshold": REACH_POSITION_SUCCESS_THRESHOLD,
            "rotation_threshold": REACH_ROTATION_SUCCESS_THRESHOLD,
            "pose_range": OBJECT_POSE_RANGE,
            "velocity_range": OBJECT_VELOCITY_RANGE,
        },
    )

@configclass
class GraspStageEventCfg(ReachStageEventCfg):
    # Disable reach-based reset
    reset_object_on_success = None

@configclass
class PickPlaceEventCfg:
    """Event configuration for resets and randomization."""
    
    reset_object_on_success = EventTermCfg(
        func=mdp.reset_object_on_success,
        mode="interval",
        interval_range_s=(0, 0), # Check every step
        params={
            "object_cfg": SceneEntityCfg("object"),
            "source_frame_cfg": SceneEntityCfg("ee_frame"),
            "target_frame_cfg": SceneEntityCfg("hover_target_frame"),
            "position_threshold": REACH_POSITION_SUCCESS_THRESHOLD,
            "rotation_threshold": REACH_ROTATION_SUCCESS_THRESHOLD,
            "pose_range": OBJECT_POSE_RANGE,
            "velocity_range": OBJECT_VELOCITY_RANGE,
        },
    )
    
    # Reset object and target poses
    reset_object = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": OBJECT_POSE_RANGE,
            "velocity_range": OBJECT_VELOCITY_RANGE,
        },
    )

    reset_target = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("target"),
            "pose_range": {
                "x": (-0.25, 0.25),      # 0.0 ± 0.25 = [-0.25, 0.25]
                "y": (0.25, -0.25),      # -0.45 ± 0.25 = [-0.7, -0.2]
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # Reset robot to home position
    reset_articulation_to_default = EventTermCfg(
        func=mdp.reset_articulation_to_default,
        mode="reset",
    )

    randomize_object_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object", body_ids=[0]),  # RigidObject has only one body
            "mass_distribution_params": (0.01, 0.5),  # 10g to 500g
            "operation": "abs",  # Set absolute mass values
            "distribution": "uniform",
        },
    )

    # This event term randomizes the scale of the cube.
    # The mode is set to 'prestartup', which means that the scale is randomize on the USD stage before the
    # simulation starts.
    # Note: USD-level randomizations require the flag 'replicate_physics' to be set to False.
    # Base size: 2.5cm  (Range: 1cm to 4.5cm)
    randomize_object_scale = EventTermCfg(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",  # Must be at startup, not reset. Every environment must keep its scale!
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "scale_range": {
                "x": (0.4, 1.8),   # 2.5cm * 0.4 = 1cm, 2.5cm * 1.8 = 4.5cm
                "y": (0.4, 1.8),
                "z": (0.4, 1.8),
            },
        },
    )

    # This event term randomizes the visual color of the cube.
    # Similar to the scale randomization, this is also a USD-level randomization and requires the flag
    # 'replicate_physics' to be set to False.
    randomize_color = EventTermCfg(
        func=mdp.randomize_visual_color,
        mode="prestartup",
        params={
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "asset_cfg": SceneEntityCfg("object"),
            "mesh_name": "geometry/mesh",
            "event_name": "rep_cube_randomize_color",
        },
    )

    # Randomize robot joint stiffness and damping
    robot_joint_stiffness_and_damping = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=200,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.75, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Randomize robot joint friction
    # joint_friction = EventTermCfg(
    #     func=mdp.randomize_joint_parameters,
    #     min_step_count_between_reset=200,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "friction_distribution_params": (0.0, 0.1),
    #         "operation": "add",
    #         "distribution": "uniform",
    #     },
    # )

    # Custom logging
    log_custom_metrics = EventTermCfg(
        func=mdp.log_custom_metrics,
        mode="reset",
    )
