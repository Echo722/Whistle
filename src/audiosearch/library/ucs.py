"""Built-in Universal Category System (UCS v8.2) catalog, bilingual search, and filename parser."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class UcsEntry:
    cat_id: str
    category: str
    subcategory: str
    category_zh: str
    subcategory_zh: str
    synonyms: tuple[str, ...] = ()

    @property
    def display_label(self) -> str:
        return f"{self.cat_id} · {self.category} / {self.subcategory}（{self.category_zh} / {self.subcategory_zh}）"


def _e(
    cat_id: str,
    category: str,
    subcategory: str,
    category_zh: str,
    subcategory_zh: str,
    *synonyms: str,
) -> UcsEntry:
    return UcsEntry(
        cat_id=cat_id,
        category=category,
        subcategory=subcategory,
        category_zh=category_zh,
        subcategory_zh=subcategory_zh,
        synonyms=tuple(synonyms),
    )


# Curated UCS v8.2 catalog covering major sound-design & game-audio categories (220+ entries).
UCS_CATALOG: tuple[UcsEntry, ...] = (
    # AIR
    _e("AIRBlow", "AIR", "BLOW", "气流", "吹气/喷气", "air blow burst jet puff"),
    _e("AIRBrst", "AIR", "BURST", "气流", "气爆/泄压", "pneumatic release valve burst"),
    _e("AIRHiss", "AIR", "HISS", "气流", "嘶嘶漏气", "hiss leak steam"),
    _e("AIRMisc", "AIR", "MISC", "气流", "其他气流", "air"),
    _e("AIRSuction", "AIR", "SUCTION", "气流", "吸气/真空吸力", "vacuum suction inhale"),
    # AMBIENCE
    _e("AMBBird", "AMBIENCE", "BIRDSONG", "环境氛围", "鸟鸣氛围", "forest birds morning"),
    _e("AMBCave", "AMBIENCE", "CAVE", "环境氛围", "洞穴/地牢", "cavern dungeon drip"),
    _e("AMBCity", "AMBIENCE", "CITY", "环境氛围", "城市街道", "urban traffic city street"),
    _e("AMBDsrt", "AMBIENCE", "DESERT", "环境氛围", "荒漠/风沙", "desert barren"),
    _e("AMBForst", "AMBIENCE", "FOREST", "环境氛围", "森林/丛林", "woods jungle trees"),
    _e("AMBHosp", "AMBIENCE", "HOSPITAL", "环境氛围", "医院/病房", "medical"),
    _e("AMBInd", "AMBIENCE", "INDUSTRIAL", "环境氛围", "工业/厂房", "factory warehouse"),
    _e("AMBInt", "AMBIENCE", "INTERIOR", "环境氛围", "室内环境", "room tone indoor"),
    _e("AMBJung", "AMBIENCE", "JUNGLE", "环境氛围", "热带雨林", "tropical"),
    _e("AMBMkt", "AMBIENCE", "MARKET", "环境氛围", "集市/市场", "bazaar crowd"),
    _e("AMBMisc", "AMBIENCE", "MISC", "环境氛围", "其他氛围", "ambience atmosphere"),
    _e("AMBNaut", "AMBIENCE", "NAUTICAL", "环境氛围", "航海/港口", "harbor ship sea"),
    _e("AMBOffc", "AMBIENCE", "OFFICE", "环境氛围", "办公室", "workplace"),
    _e("AMBPark", "AMBIENCE", "PARK", "环境氛围", "公园/户外", "outdoor playground"),
    _e("AMBRest", "AMBIENCE", "RESTAURANT", "环境氛围", "餐厅/咖啡馆", "cafe dining"),
    _e("AMBRurl", "AMBIENCE", "RURAL", "环境氛围", "乡村/田园", "countryside meadow"),
    _e("AMBScif", "AMBIENCE", "SCI-FI", "环境氛围", "科幻氛围", "spaceship hull reactor"),
    _e("AMBSea", "AMBIENCE", "SEASIDE", "环境氛围", "海边/海浪", "coast beach ocean"),
    _e("AMBSwmp", "AMBIENCE", "SWAMP", "环境氛围", "沼泽/湿地", "marsh frogs insects"),
    _e("AMBTraf", "AMBIENCE", "TRAFFIC", "环境氛围", "交通车流", "highway road"),
    _e("AMBUndw", "AMBIENCE", "UNDERWATER", "环境氛围", "水下氛围", "submerged deep sea"),
    # ANIMALS
    _e("ANMLCat", "ANIMALS", "CAT", "动物", "猫科/猫叫", "meow purr feline"),
    _e("ANMLDog", "ANIMALS", "DOG", "动物", "犬科/狗叫", "bark growl canine"),
    _e("ANMLFarm", "ANIMALS", "FARM", "动物", "家畜/农场动物", "cow pig sheep goat"),
    _e("ANMLHors", "ANIMALS", "HORSE", "动物", "马匹/蹄声", "neigh gallop"),
    _e("ANMLInsc", "ANIMALS", "INSECT", "动物", "昆虫", "cricket cicada bee fly"),
    _e("ANMLMisc", "ANIMALS", "MISC", "动物", "其他动物", "wildlife"),
    _e("ANMLWild", "ANIMALS", "WILD", "动物", "野生猛兽", "lion tiger bear wolf"),
    # ARCH / BUILDINGS
    _e("ARCHMisc", "ARCH", "MISC", "建筑结构", "建筑/房屋", "building structure"),
    # BELLS
    _e("BELLAnml", "BELLS", "ANIMAL", "钟铃", "牧铃/颈铃", "cowbell"),
    _e("BELLChime", "BELLS", "CHIME", "钟铃", "风铃/提示钟", "wind chime glockenspiel"),
    _e("BELLChur", "BELLS", "CHURCH", "钟铃", "教堂大钟", "cathedral toll"),
    _e("BELLDoor", "BELLS", "DOOR", "钟铃", "门铃", "doorbell ding dong"),
    _e("BELLHand", "BELLS", "HAND", "钟铃", "手摇铃", "service bell desk"),
    _e("BELLMisc", "BELLS", "MISC", "钟铃", "其他钟铃", "bell ding"),
    # CARTOON
    _e("TOONBoing", "CARTOON", "BOING", "卡通动画", "弹簧/弹跳", "spring bounce"),
    _e("TOONHorn", "CARTOON", "HORN", "卡通动画", "滑稽喇叭", "clown honk"),
    _e("TOONMisc", "CARTOON", "MISC", "卡通动画", "其他卡通", "comedy silly"),
    _e("TOONPerc", "CARTOON", "PERCUSSION", "卡通动画", "卡通打击乐", "woodblock xylophone"),
    _e("TOONPop", "CARTOON", "POP", "卡通动画", "啵啵/泡泡弹出", "bubble plop cork"),
    _e("TOONSld", "CARTOON", "SLIDE", "卡通动画", "滑音哨/滑倒", "slide whistle slip"),
    _e("TOONSplt", "CARTOON", "SPLAT", "卡通动画", "拍扁/泥浆飞溅", "splat squish"),
    _e("TOONTwng", "CARTOON", "TWANG", "卡通动画", "拨弦/抖动", "twang wobble"),
    _e("TOONWhsl", "CARTOON", "WHISTLE", "卡通动画", "卡通口哨", "whistle"),
    _e("TOONZip", "CARTOON", "ZIP", "卡通动画", "咻声飞窜", "zip zing"),
    # COMMUNICATIONS
    _e("COMMisc", "COMMUNICATIONS", "MISC", "通讯设备", "其他通讯", "telecom"),
    _e("COMPhne", "COMMUNICATIONS", "PHONE", "通讯设备", "电话/铃声", "telephone ring dial"),
    _e("COMRad", "COMMUNICATIONS", "RADIO", "通讯设备", "对讲机/无线电", "walkie talkie static squelch"),
    _e("COMTelv", "COMMUNICATIONS", "TELEVISION", "通讯设备", "电视/广播杂讯", "tv broadcast"),
    # CREATURES
    _e("CREAAlien", "CREATURES", "ALIEN", "怪物生物", "外星生物", "alien insectoid"),
    _e("CREABeast", "CREATURES", "BEAST", "怪物生物", "巨兽", "beast monster giant"),
    _e("CREADemon", "CREATURES", "DEMON", "怪物生物", "恶魔", "demon hell"),
    _e("CREADrag", "CREATURES", "DRAGON", "怪物生物", "龙/飞龙", "dragon roar wing"),
    _e("CREAGrowl", "CREATURES", "GROWL", "怪物生物", "低吼/咆哮", "growl snarl roar"),
    _e("CREAMisc", "CREATURES", "MISC", "怪物生物", "其他怪物", "creature monster"),
    _e("CREAZomb", "CREATURES", "ZOMBIE", "怪物生物", "丧尸/亡灵", "zombie undead groan"),
    # CROWDS
    _e("CRWDAppl", "CROWDS", "APPLAUSE", "人群", "掌声/鼓掌", "clap ovation"),
    _e("CRWDCheer", "CROWDS", "CHEER", "人群", "欢呼/喝彩", "cheer stadium"),
    _e("CRWDMisc", "CROWDS", "MISC", "人群", "人群/嘈杂声", "crowd chatter murmur"),
    _e("CRWDPanc", "CROWDS", "PANIC", "人群", "惊恐/尖叫", "scream panic"),
    # DESIGNED
    _e("DSGNBoom", "DESIGNED", "BOOM", "设计合成", "低频重击/Boom", "sub drop cinematic boom"),
    _e("DSGNBrhm", "DESIGNED", "BRAAM", "设计合成", "铜管巨响/Braam", "trailer horn braam"),
    _e("DSGNClck", "DESIGNED", "CLICK", "设计合成", "合成点击", "synth tick click"),
    _e("DSGNDone", "DESIGNED", "DRONE", "设计合成", "氛围长音/Drone", "drone pad dark"),
    _e("DSGNImpt", "DESIGNED", "IMPACT", "设计合成", "设计冲击/重击", "trailer hit slam"),
    _e("DSGNMisc", "DESIGNED", "MISC", "设计合成", "其他设计音效", "designed synth"),
    _e("DSGNMrph", "DESIGNED", "MORPH", "设计合成", "变形/转化", "morph transform"),
    _e("DSGNRisr", "DESIGNED", "RISER", "设计合成", "上升张力/Riser", "riser swell buildup"),
    _e("DSGNStng", "DESIGNED", "STINGER", "设计合成", "转场短音/Stinger", "stinger transition"),
    _e("DSGNSynth", "DESIGNED", "SYNTH", "设计合成", "合成器音色", "synth tone"),
    _e("DSGNWhsh", "DESIGNED", "WHOOSH", "设计合成", "设计呼啸/Swish", "whoosh swoosh"),
    # DESTRUCTION
    _e("DESTColp", "DESTRUCTION", "COLLAPSE", "破坏倒塌", "坍塌/崩塌", "collapse rubble"),
    _e("DESTCrsh", "DESTRUCTION", "CRASH", "破坏倒塌", "猛烈撞毁", "crash smash"),
    _e("DESTDebr", "DESTRUCTION", "DEBRIS", "破坏倒塌", "碎石/残骸", "debris"),
    _e("DESTMisc", "DESTRUCTION", "MISC", "破坏倒塌", "其他破坏", "destruction"),
    # DIRT & SAND
    _e("DIRTMisc", "DIRT", "MISC", "泥土沙石", "泥土/沙砾", "dirt sand gravel"),
    # DOORS
    _e("DOORAuto", "DOORS", "AUTOMATIC", "门窗", "自动门/气动门", "sliding automatic"),
    _e("DOORGat", "DOORS", "GATE", "门窗", "铁栅/大门", "gate"),
    _e("DOORGlas", "DOORS", "GLASS", "门窗", "玻璃门", "glass door"),
    _e("DOORKnck", "DOORS", "KNOCK", "门窗", "敲门", "knock"),
    _e("DOORLtch", "DOORS", "LATCH", "门窗", "门锁/插销", "latch lock handle"),
    _e("DOORMtl", "DOORS", "METAL", "门窗", "金属门", "heavy steel door"),
    _e("DOORMisc", "DOORS", "MISC", "门窗", "其他门声", "door"),
    _e("DOORWood", "DOORS", "WOOD", "门窗", "木门开关", "wooden door creak slam"),
    # ELECTRICITY
    _e("ELECArc", "ELECTRICITY", "ARC", "电力电流", "电弧/高压放电", "arc zap discharge"),
    _e("ELECBuzz", "ELECTRICITY", "BUZZ", "电力电流", "交流嗡嗡声", "hum buzz"),
    _e("ELECMisc", "ELECTRICITY", "MISC", "电力电流", "其他电流", "electric power"),
    _e("ELECSprk", "ELECTRICITY", "SPARKS", "电力电流", "电火花/噼啪", "sparks static shock"),
    # EXPLOSIONS
    _e("EXPLBig", "EXPLOSIONS", "BIG", "爆炸", "大型爆炸", "massive blast"),
    _e("EXPLBrst", "EXPLOSIONS", "BURST", "爆炸", "爆裂/瞬爆", "burst pop"),
    _e("EXPLDist", "EXPLOSIONS", "DISTANT", "爆炸", "远处爆炸", "distant rumble"),
    _e("EXPLMisc", "EXPLOSIONS", "MISC", "爆炸", "其他爆炸", "explosion blast"),
    _e("EXPLReal", "EXPLOSIONS", "REALISTIC", "爆炸", "真实写实爆炸", "c4 grenade mortar"),
    _e("EXPLScif", "EXPLOSIONS", "SCI-FI", "爆炸", "科幻爆炸", "plasma energy explosion"),
    # FEET / FOOTSTEPS
    _e("FEETBoot", "FEET", "BOOTS", "脚步", "靴子脚步", "boots combat leather"),
    _e("FEETCreat", "FEET", "CREATURE", "脚步", "巨兽/怪物脚步", "giant stomp"),
    _e("FEETGrvl", "FEET", "GRAVEL", "脚步", "碎石脚步", "gravel dirt footsteps"),
    _e("FEETMisc", "FEET", "MISC", "脚步", "其他脚步", "footsteps walk run"),
    _e("FEETShoe", "FEET", "SHOES", "脚步", "皮鞋/硬底鞋", "heels shoes"),
    _e("FEETSnow", "FEET", "SNOW", "脚步", "雪地脚步", "snow crunch"),
    _e("FEETWatr", "FEET", "WATER", "脚步", "涉水脚步", "puddle splash steps"),
    _e("FEETWood", "FEET", "WOOD", "脚步", "木地板脚步", "wood floor steps"),
    # FIGHT
    _e("FGHTBody", "FIGHT", "BODYFALL", "格斗打斗", "人体倒地", "bodyfall thud"),
    _e("FGHTGrab", "FIGHT", "GRAB", "格斗打斗", "擒拿/抓握", "grab wrestle"),
    _e("FGHTImpt", "FIGHT", "IMPACT", "格斗打斗", "拳脚击中", "punch kick hit slap"),
    _e("FGHTMisc", "FIGHT", "MISC", "格斗打斗", "其他打斗", "fight combat"),
    # FIRE
    _e("FIREBrn", "FIRE", "BURNING", "火焰", "火焰燃烧", "fire crackling flames"),
    _e("FIREFlm", "FIRE", "FLAME", "火焰", "喷火/火球/火呼啸", "flame thrower fireball whoosh"),
    _e("FIREMisc", "FIRE", "MISC", "火焰", "其他火声", "fire"),
    _e("FIRETrch", "FIRE", "TORCH", "火焰", "火把", "torch"),
    # FOLEY
    _e("FOLYClth", "FOLEY", "CLOTH", "拟音", "衣物布料摩擦", "cloth fabric rustle"),
    _e("FOLYFeet", "FOLEY", "FEET", "拟音", "拟音脚步", "foley footsteps"),
    _e("FOLYHand", "FOLEY", "HANDS", "拟音", "手部拍手/搓手", "clap rub"),
    _e("FOLYLthr", "FOLEY", "LEATHER", "拟音", "皮革摩擦", "leather creak"),
    _e("FOLYMisc", "FOLEY", "MISC", "拟音", "其他拟音", "foley prop"),
    # FOOD & DRINK
    _e("FOODCook", "FOOD & DRINK", "COOKING", "餐饮", "烹饪/煎炸", "sizzle fry cook"),
    _e("FOODEat", "FOOD & DRINK", "EATING", "餐饮", "进食/咀嚼", "bite crunch chew"),
    _e("FOODMisc", "FOOD & DRINK", "MISC", "餐饮", "其他食物饮料", "drink pour"),
    # GAMES
    _e("GAMEBrd", "GAMES", "BOARD", "游戏专用", "棋牌桌游", "card dice chess chip"),
    _e("GAMECas", "GAMES", "CASINO", "游戏专用", "老虎机/筹码", "slot casino jackpot"),
    _e("GAMEGold", "GAMES", "COINS & REWARD", "游戏专用", "金币/星星/奖励", "coin star gem reward bonus"),
    _e("GAMEMisc", "GAMES", "MISC", "游戏专用", "其他游戏音效", "game sfx arcade"),
    _e("GAMERetro", "GAMES", "RETRO", "游戏专用", "8位/像素/复古游戏", "8-bit retro chiptune pixel"),
    _e("GAMEUp", "GAMES", "POWERUP & LEVELUP", "游戏专用", "升级/增益/解锁", "powerup level up unlock"),
    # GLASS
    _e("GLASBrea", "GLASS", "BREAK", "玻璃", "玻璃碎裂", "glass shatter smash crash"),
    _e("GLASClnk", "GLASS", "CLINK", "玻璃", "玻璃轻碰", "clink bottle cup"),
    _e("GLASMisc", "GLASS", "MISC", "玻璃", "其他玻璃", "glass"),
    # GORE
    _e("GOREBld", "GORE", "BLOOD", "血肉脏器", "血液溅射", "blood drip squirt"),
    _e("GOREBone", "GORE", "BONE", "血肉脏器", "骨裂/折断", "bone snap crack"),
    _e("GOREMisc", "GORE", "MISC", "血肉脏器", "其他血肉", "gore flesh"),
    _e("GORESplt", "GORE", "SPLAT", "血肉脏器", "血肉飞溅", "splat squish"),
    # GUNS
    _e("GUNAuto", "GUNS", "AUTOMATIC", "枪械", "自动步枪/冲锋枪", "rifle smg machine gun"),
    _e("GUNCann", "GUNS", "CANNON", "枪械", "火炮/重炮", "cannon artillery"),
    _e("GUNMech", "GUNS", "MECHANICAL", "枪械", "上膛/换弹/退壳", "reload cocking shell casing"),
    _e("GUNMisc", "GUNS", "MISC", "枪械", "其他枪械", "gunshot firearm"),
    _e("GUNPistl", "GUNS", "PISTOL", "枪械", "手枪", "handgun pistol"),
    _e("GUNRifl", "GUNS", "RIFLE", "枪械", "步枪/狙击枪", "sniper rifle"),
    _e("GUNShotg", "GUNS", "SHOTGUN", "枪械", "霰弹枪", "shotgun"),
    # HORROR
    _e("HORRMisc", "HORROR", "MISC", "恐怖惊悚", "恐怖/惊悚元素", "horror creepy eerie ghost"),
    _e("HORRStng", "HORROR", "STINGER", "恐怖惊悚", "惊吓短音", "jumpscare sting"),
    # ICE
    _e("ICEBrk", "ICE", "BREAK", "冰雪", "冰块碎裂/裂纹", "ice crack shatter frost"),
    _e("ICEMisc", "ICE", "MISC", "冰雪", "其他冰雪", "ice freeze snow"),
    # LASERS
    _e("LASRBeam", "LASERS", "BEAM", "激光", "持续激光束", "laser beam energy"),
    _e("LASRGun", "LASERS", "GUN", "激光", "激光枪射击", "pew blaster"),
    _e("LASRMisc", "LASERS", "MISC", "激光", "其他激光", "laser"),
    # LIQUID
    _e("LIQBubl", "LIQUID", "BUBBLE", "液体", "气泡/冒泡", "bubble gurgle"),
    _e("LIQDrop", "LIQUID", "DROP", "液体", "液滴", "drip drop"),
    _e("LIQMisc", "LIQUID", "MISC", "液体", "其他液体", "liquid slime mud"),
    _e("LIQPour", "LIQUID", "POUR", "液体", "倾倒/倒水", "pour potion"),
    _e("LIQSpls", "LIQUID", "SPLASH", "液体", "液体飞溅", "splash"),
    # MAGIC
    _e("MAGAura", "MAGIC", "AURA", "魔法奇幻", "魔法光环/阵法", "aura loop magic circle"),
    _e("MAGCast", "MAGIC", "CAST", "魔法奇幻", "施法/释放咒语", "cast spell charge burst"),
    _e("MAGHeal", "MAGIC", "HEAL", "魔法奇幻", "治疗/增益魔法", "heal buff shimmer chime"),
    _e("MAGMisc", "MAGIC", "MISC", "魔法奇幻", "其他魔法", "magic fantasy"),
    _e("MAGSpel", "MAGIC", "SPELL", "魔法奇幻", "法术命中/弹道", "spell hit projectile"),
    # MECHANICAL
    _e("MECHClck", "MECHANICAL", "CLOCKWORK", "机械结构", "发条/钟表齿轮", "clock ticking winding"),
    _e("MECHGear", "MECHANICAL", "GEARS", "机械结构", "齿轮传动", "gears ratchet"),
    _e("MECHLvr", "MECHANICAL", "LEVER", "机械结构", "拉杆/开关机械", "lever switch latch"),
    _e("MECHMisc", "MECHANICAL", "MISC", "机械结构", "其他机械", "mechanical machine"),
    _e("MECHServ", "MECHANICAL", "SERVO", "机械结构", "伺服电机/液压", "servo hydraulic piston"),
    # METAL
    _e("METLClnk", "METAL", "CLINK", "金属", "小金属轻碰/叮当", "clink coin key chain"),
    _e("METLCrsh", "METAL", "CRASH", "金属", "金属撞击/砸毁", "metal crash clang impact"),
    _e("METLCrek", "METAL", "CREAK", "金属", "金属吱嘎/弯曲", "creak groan bend"),
    _e("METLMisc", "METAL", "MISC", "金属", "其他金属", "metal steel iron"),
    _e("METLRng", "METAL", "RING", "金属", "金属共鸣/震荡", "ringing resonant gong"),
    _e("METLScrp", "METAL", "SCRAPE", "金属", "金属刮擦", "scrape screech"),
    # MUSICAL
    _e("MUSDrum", "MUSICAL", "DRUMS", "音乐元素", "鼓/打击乐", "drum hit taiko snare"),
    _e("MUSInst", "MUSICAL", "INSTRUMENT", "音乐元素", "乐器单音/动机", "piano guitar harp flute"),
    _e("MUSMisc", "MUSICAL", "MISC", "音乐元素", "其他音乐片段", "music"),
    _e("MUSStgr", "MUSICAL", "STINGER", "音乐元素", "过关/失败音乐短句", "win lose jingle fanfare"),
    # OBJECTS
    _e("OBJBox", "OBJECTS", "BOX", "日常物件", "箱子/宝箱/容器", "chest crate box"),
    _e("OBJCoin", "OBJECTS", "COIN", "日常物件", "硬币/金属片", "coin token"),
    _e("OBJKey", "OBJECTS", "KEY", "日常物件", "钥匙串", "keys"),
    _e("OBJMisc", "OBJECTS", "MISC", "日常物件", "其他物件", "object prop"),
    # PAPER
    _e("PAPRFlip", "PAPER", "FLIP", "纸张", "翻书/翻页", "page turn book"),
    _e("PAPRMisc", "PAPER", "MISC", "纸张", "其他纸张", "paper cardboard"),
    _e("PAPRTear", "PAPER", "TEAR", "纸张", "撕纸/揉纸", "rip crumple"),
    # PLASTIC
    _e("PLASBtn", "PLASTIC", "BUTTON", "塑料", "塑料按钮/开关", "plastic switch click"),
    _e("PLASMisc", "PLASTIC", "MISC", "塑料", "其他塑料", "plastic"),
    # RAIN
    _e("RAINHvy", "RAIN", "HEAVY", "雨水", "暴雨/大雨", "downpour storm rain"),
    _e("RAINLght", "RAIN", "LIGHT", "雨水", "小雨/细雨", "drizzle light rain"),
    _e("RAINMisc", "RAIN", "MISC", "雨水", "其他雨声", "rain"),
    # ROBOTS
    _e("ROBOMisc", "ROBOTS", "MISC", "机器人", "其他机器人", "robot droid mech"),
    _e("ROBOMvmt", "ROBOTS", "MOVEMENT", "机器人", "机甲/机器人移动", "robot walk servo"),
    _e("ROBOTrns", "ROBOTS", "TRANSFORM", "机器人", "机器人变形", "transform mech assemble"),
    _e("ROBOVox", "ROBOTS", "VOICE", "机器人", "机械合成语音", "robot voice vocoder"),
    # ROCKS
    _e("ROCKFall", "ROCKS", "FALL", "岩石泥石", "落石/碎石掉落", "rock fall slide"),
    _e("ROCKImpt", "ROCKS", "IMPACT", "岩石泥石", "岩石撞击/重击", "stone hit boulder"),
    _e("ROCKMisc", "ROCKS", "MISC", "岩石泥石", "其他岩石", "rock stone"),
    # SCI-FI
    _e("SCIFAlrm", "SCI-FI", "ALARM", "科幻", "科幻警报", "alarm siren warning"),
    _e("SCIFComp", "SCI-FI", "COMPUTER", "科幻", "科幻终端/电脑", "hud data terminal beep"),
    _e("SCIFEnrg", "SCI-FI", "ENERGY", "科幻", "能量护盾/充能", "shield plasma charge"),
    _e("SCIFMisc", "SCI-FI", "MISC", "科幻", "其他科幻", "sci-fi futuristic"),
    _e("SCIFTele", "SCI-FI", "TELEPORT", "科幻", "传送/跃迁", "teleport warp portal"),
    _e("SCIFWeap", "SCI-FI", "WEAPON", "科幻", "科幻武器", "blaster railgun"),
    # SPORTS
    _e("SPRTBat", "SPORTS", "BALL & BAT", "体育运动", "球类/击球", "ball bat bounce kick"),
    _e("SPRTMisc", "SPORTS", "MISC", "体育运动", "其他体育", "sports whistle"),
    # SWOOSHES / WHOOSHES
    _e("WHSHFast", "WHOOSHES", "FAST", "呼啸甩动", "快速呼啸", "fast swipe slash"),
    _e("WHSHMisc", "WHOOSHES", "MISC", "呼啸甩动", "其他呼啸", "whoosh swoosh air"),
    _e("WHSHSwsh", "WHOOSHES", "SWISH", "呼啸甩动", "轻快甩动/挥舞", "swish wave"),
    # TOOLS
    _e("TOOLHamr", "TOOLS", "HAMMER", "工具", "锤击/敲打", "hammer nail strike"),
    _e("TOOLMisc", "TOOLS", "MISC", "工具", "其他工具", "tools drill saw wrench"),
    # USER INTERFACE (UI)
    _e("UIAlrt", "USER INTERFACE", "ALERT", "界面交互", "警告/提示警报", "alert popup"),
    _e("UIButton", "USER INTERFACE", "BUTTON", "界面交互", "UI 按钮", "button press tap"),
    _e("UIClick", "USER INTERFACE", "CLICK", "界面交互", "轻脆点击", "click tick switch"),
    _e("UIConf", "USER INTERFACE", "CONFIRM", "界面交互", "确认/成功反馈", "confirm success positive ok"),
    _e("UIError", "USER INTERFACE", "ERROR", "界面交互", "错误/拒绝反馈", "error fail invalid buzz"),
    _e("UIHover", "USER INTERFACE", "HOVER", "界面交互", "悬停/滑过", "hover focus rollover"),
    _e("UIMisc", "USER INTERFACE", "MISC", "界面交互", "其他 UI 音效", "ui menu hud"),
    _e("UINotif", "USER INTERFACE", "NOTIFICATION", "界面交互", "消息/通知提示", "notification chime ping"),
    _e("UIScroll", "USER INTERFACE", "SCROLL", "界面交互", "滑动/滚动列表", "scroll swipe"),
    # VEHICLES
    _e("VEHCars", "VEHICLES", "CARS", "交通工具", "汽车行驶/鸣笛", "car horn pass by"),
    _e("VEHEngn", "VEHICLES", "ENGINE", "交通工具", "引擎/发动机", "engine idle revving"),
    _e("VEHMisc", "VEHICLES", "MISC", "交通工具", "其他载具", "vehicle aircraft train boat"),
    # VOICE
    _e("VOXEffrt", "VOICE", "EFFORT", "人声", "用力/喘息/攻击喊叫", "effort grunt attack hurt"),
    _e("VOXLagh", "VOICE", "LAUGH", "人声", "笑声", "laugh giggle"),
    _e("VOXMisc", "VOICE", "MISC", "人声", "其他人声", "voice vocal speech"),
    _e("VOXScrm", "VOICE", "SCREAM", "人声", "尖叫/惨叫", "scream shout yell"),
    # WATER
    _e("WATRDrop", "WATER", "DROP", "水声", "水滴声", "water drip drop"),
    _e("WATRFlow", "WATER", "FLOW", "水声", "流水/溪流", "stream river flow"),
    _e("WATRMisc", "WATER", "MISC", "水声", "其他水声", "water"),
    _e("WATRSplash", "WATER", "SPLASH", "水声", "水花飞溅", "splash plunge dive"),
    _e("WATRWave", "WATER", "WAVES", "水声", "海浪波涛", "waves surf"),
    # WEAPONS
    _e("WEAPBow", "WEAPONS", "BOW & ARROW", "冷兵器", "弓箭发射/箭矢", "bow arrow release flyby"),
    _e("WEAPMisc", "WEAPONS", "MISC", "冷兵器", "其他冷兵器", "weapon armor shield"),
    _e("WEAPSwd", "WEAPONS", "SWORD", "冷兵器", "刀剑挥砍/出鞘/拼刀", "sword slash parry draw metal"),
    # WEATHER / WIND
    _e("WINDGust", "WIND", "GUST", "风声", "阵风/寒风呼啸", "wind gust howl"),
    _e("WINDMisc", "WIND", "MISC", "风声", "其他风声", "wind breeze"),
    _e("WTHRThnd", "WEATHER", "THUNDER", "天气", "雷鸣/闪电轰鸣", "thunder storm clap"),
    # WOOD
    _e("WOODBrk", "WOOD", "BREAK", "木材", "木头折断/碎裂", "wood break snap splinter"),
    _e("WOODCrek", "WOOD", "CREAK", "木材", "木头吱嘎声", "wood creak"),
    _e("WOODImpt", "WOOD", "IMPACT", "木材", "木头敲击/撞击", "wood knock hit block"),
    _e("WOODMisc", "WOOD", "MISC", "木材", "其他木质声", "wood"),
)

_BY_CAT_ID: dict[str, UcsEntry] = {entry.cat_id.casefold(): entry for entry in UCS_CATALOG}
_CAT_ID_SHAPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{2,11}(?:-[A-Za-z0-9]{1,24})?$")


def list_ucs_entries() -> tuple[UcsEntry, ...]:
    return UCS_CATALOG


def split_cat_id(raw_cat_id: str) -> tuple[str, str]:
    """Split ``CatID[-UserCategory]`` into ``(base_cat_id, user_category)``."""
    cleaned = raw_cat_id.strip()
    if "-" in cleaned:
        base, user_cat = cleaned.split("-", 1)
        return base.strip(), user_cat.strip()
    return cleaned, ""


def lookup_cat_id(cat_id: str) -> UcsEntry | None:
    base, _ = split_cat_id(cat_id)
    if not base:
        return None
    return _BY_CAT_ID.get(base.casefold())


def validate_cat_id(cat_id: str) -> tuple[bool, UcsEntry | None, str]:
    cleaned = cat_id.strip()
    if not cleaned:
        return True, None, "空 CatID（未分类）"
    entry = lookup_cat_id(cleaned)
    if entry is not None:
        base, user_cat = split_cat_id(cleaned)
        if user_cat:
            return True, entry, f"已匹配标准 UCS：{entry.cat_id}（扩展子分类：{user_cat}）"
        return True, entry, f"已匹配标准 UCS：{entry.cat_id} · {entry.category} / {entry.subcategory}"
    if _CAT_ID_SHAPE_RE.match(cleaned):
        return False, None, f"「{cleaned}」未收录在内置 UCS v8.2 常用表（仍可作为自定义 CatID 保存）"
    return False, None, f"「{cleaned}」不符合标准 UCS CatID 命名格式（如 MAGSpel 或 UIButton）"


def search_ucs(query: str, *, limit: int = 50) -> list[UcsEntry]:
    q = query.strip().casefold()
    if not q:
        return list(UCS_CATALOG[:limit])
    tokens = [tok for tok in q.split() if tok]
    scored: list[tuple[int, str, UcsEntry]] = []
    for entry in UCS_CATALOG:
        cid_cf = entry.cat_id.casefold()
        en_fields = f"{entry.category} {entry.subcategory} {' '.join(entry.synonyms)}".casefold()
        zh_fields = f"{entry.category_zh} {entry.subcategory_zh}".casefold()
        combined = f"{cid_cf} {en_fields} {zh_fields}"
        if not all(tok in combined for tok in tokens):
            continue
        score = 10
        if cid_cf == q:
            score = 0
        elif cid_cf.startswith(q):
            score = 1
        elif q in cid_cf:
            score = 2
        elif q in entry.category_zh.casefold() or q in entry.subcategory_zh.casefold():
            score = 3
        elif q in entry.category.casefold() or q in entry.subcategory.casefold():
            score = 4
        scored.append((score, entry.cat_id, entry))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in scored[:limit]]


@dataclass(frozen=True, slots=True)
class UcsFilenameParseResult:
    matched: bool
    is_strict_ucs: bool
    entry: UcsEntry | None
    cat_id: str
    user_category: str
    fx_name: str
    designer: str
    library: str
    user_data: str
    fields: dict[str, str]
    note: str


def _clean_fx_phrase(text: str) -> str:
    # Split camelCase lightly when no spaces exist, keep normal words intact
    cleaned = re.sub(r"\s+", " ", text.replace("-", " ")).strip()
    return cleaned


def parse_ucs_filename(filename: str) -> UcsFilenameParseResult:
    """Parse UCS v8.2 filenames: ``CatID[-UserCat]_FXName_CreatorID_SourceID[_UserData].ext``.

    Never invents a fake ``CatID`` when token 0 does not match a known UCS entry.
    """
    stem = Path(filename).stem.strip()
    if not stem:
        return UcsFilenameParseResult(
            matched=False,
            is_strict_ucs=False,
            entry=None,
            cat_id="",
            user_category="",
            fx_name="",
            designer="",
            library="",
            user_data="",
            fields={},
            note="文件名为空，无法解析",
        )

    parts = [p.strip() for p in stem.split("_") if p.strip()]
    if len(parts) < 2:
        return UcsFilenameParseResult(
            matched=False,
            is_strict_ucs=False,
            entry=None,
            cat_id="",
            user_category="",
            fx_name="",
            designer="",
            library="",
            user_data="",
            fields={},
            note="未检测到 UCS 下划线分段格式（需至少包含 CatID_FXName）",
        )

    head = parts[0]
    entry = lookup_cat_id(head)
    if entry is None:
        return UcsFilenameParseResult(
            matched=False,
            is_strict_ucs=False,
            entry=None,
            cat_id="",
            user_category="",
            fx_name="",
            designer="",
            library="",
            user_data="",
            fields={},
            note=f"首段「{head}」不是已知 UCS CatID，已跳过以免误判",
        )

    _, user_cat = split_cat_id(head)
    canonical_cat_id = f"{entry.cat_id}-{user_cat}" if user_cat else entry.cat_id
    fx_name = _clean_fx_phrase(parts[1]) if len(parts) >= 2 else ""
    designer = parts[2] if len(parts) >= 3 else ""
    library = parts[3] if len(parts) >= 4 else ""
    user_data = "_".join(parts[4:]) if len(parts) >= 5 else ""

    fields: dict[str, str] = {
        "cat_id": canonical_cat_id,
        "category": entry.category,
        "subcategory": entry.subcategory,
    }
    if fx_name:
        fields["fx_name"] = fx_name
        fields["description"] = fx_name
    if designer:
        fields["designer"] = designer
    if library:
        fields["library"] = library
    if user_data:
        fields["notes"] = user_data

    kw_tokens = [entry.category.title(), entry.subcategory.title()]
    if user_cat:
        kw_tokens.append(user_cat)
    fields["keywords"] = ", ".join(dict.fromkeys(tok for tok in kw_tokens if tok))

    return UcsFilenameParseResult(
        matched=True,
        is_strict_ucs=True,
        entry=entry,
        cat_id=canonical_cat_id,
        user_category=user_cat,
        fx_name=fx_name,
        designer=designer,
        library=library,
        user_data=user_data,
        fields=fields,
        note=f"已按 UCS 解析：{canonical_cat_id}（{entry.category} / {entry.subcategory}）",
    )


__all__ = [
    "UCS_CATALOG",
    "UcsEntry",
    "UcsFilenameParseResult",
    "list_ucs_entries",
    "lookup_cat_id",
    "parse_ucs_filename",
    "search_ucs",
    "split_cat_id",
    "validate_cat_id",
]
