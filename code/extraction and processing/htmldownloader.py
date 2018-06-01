# """
# for the first 100000 blocks : 
# 	download webpage for block. 

import time
import sys
import urllib2,cookielib

for i in range(881,100000):
	
	
	site= 'https://www.etherchain.org/block/' + str(i)
	hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
	       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
	       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
	       'Accept-Encoding': 'none',
	       'Accept-Language': 'en-US,en;q=0.8',
	       'Connection': 'keep-alive'}

	req = urllib2.Request(site, headers=hdr)
	
	try:
	    page = urllib2.urlopen(req)
	except urllib2.HTTPError, e:
	    print e.fp.read()

	content = page.read()

	blocklist = str('block_' + str(i) +  '.html')
	sys.stdout=open(blocklist,"w")
	print(content)
	sys.stdout.close()
	time.sleep(7)



# """

# from selenium import webdriver 
# from selenium.webdriver.common.keys import Keys
# import pyautogui as pag
# import time
# import pandas as pd
# #blocklist = ["blockchain_0","blockchain_1","blockchain_2","blockchain_3","blockchain_4","blockchain_5","blockchain_6","blockchain_7","blockchain_8","blockchain_9","blockchain_10","blockchain_11","blockchain_12","blockchain_13","blockchain_14","blockchain_15","blockchain_16","blockchain_17","blockchain_18","blockchain_19","blockchain_20","blockchain_21","blockchain_22","blockchain_23","blockchain_24","blockchain_25","blockchain_26","blockchain_27","blockchain_28","blockchain_29","blockchain_30","blockchain_31","blockchain_32","blockchain_33","blockchain_34","blockchain_35","blockchain_36","blockchain_37","blockchain_38","blockchain_39","blockchain_40","blockchain_41","blockchain_42","blockchain_43","blockchain_44","blockchain_45","blockchain_46","blockchain_47","blockchain_48","blockchain_49","blockchain_50","blockchain_51","blockchain_52","blockchain_53","blockchain_54","blockchain_55","blockchain_56","blockchain_57","blockchain_58","blockchain_59","blockchain_60","blockchain_61","blockchain_62","blockchain_63","blockchain_64","blockchain_65","blockchain_66","blockchain_67","blockchain_68","blockchain_69","blockchain_70","blockchain_71","blockchain_72","blockchain_73","blockchain_74","blockchain_75","blockchain_76","blockchain_77","blockchain_78","blockchain_79","blockchain_80","blockchain_81","blockchain_82","blockchain_83","blockchain_84","blockchain_85","blockchain_86","blockchain_87","blockchain_88","blockchain_89","blockchain_90","blockchain_91","blockchain_92","blockchain_93","blockchain_94","blockchain_95","blockchain_96","blockchain_97","blockchain_98","blockchain_99","blockchain_100","blockchain_101","blockchain_102","blockchain_103","blockchain_104","blockchain_105","blockchain_106","blockchain_107","blockchain_108","blockchain_109","blockchain_110","blockchain_111","blockchain_112","blockchain_113","blockchain_114","blockchain_115","blockchain_116","blockchain_117","blockchain_118","blockchain_119","blockchain_120","blockchain_121","blockchain_122","blockchain_123","blockchain_124","blockchain_125","blockchain_126","blockchain_127","blockchain_128","blockchain_129","blockchain_130","blockchain_131","blockchain_132","blockchain_133","blockchain_134","blockchain_135","blockchain_136","blockchain_137","blockchain_138","blockchain_139","blockchain_140","blockchain_141","blockchain_142","blockchain_143","blockchain_144","blockchain_145","blockchain_146","blockchain_147","blockchain_148","blockchain_149","blockchain_150","blockchain_151","blockchain_152","blockchain_153","blockchain_154","blockchain_155","blockchain_156","blockchain_157","blockchain_158","blockchain_159","blockchain_160","blockchain_161","blockchain_162","blockchain_163","blockchain_164","blockchain_165","blockchain_166","blockchain_167","blockchain_168","blockchain_169","blockchain_170","blockchain_171","blockchain_172","blockchain_173","blockchain_174","blockchain_175","blockchain_176","blockchain_177","blockchain_178","blockchain_179","blockchain_180","blockchain_181","blockchain_182","blockchain_183","blockchain_184","blockchain_185","blockchain_186","blockchain_187","blockchain_188","blockchain_189","blockchain_190","blockchain_191","blockchain_192","blockchain_193","blockchain_194","blockchain_195","blockchain_196","blockchain_197","blockchain_198","blockchain_199","blockchain_200","blockchain_201","blockchain_202","blockchain_203","blockchain_204","blockchain_205","blockchain_206","blockchain_207","blockchain_208","blockchain_209","blockchain_210","blockchain_211","blockchain_212","blockchain_213","blockchain_214","blockchain_215","blockchain_216","blockchain_217","blockchain_218","blockchain_219","blockchain_220","blockchain_221","blockchain_222","blockchain_223","blockchain_224","blockchain_225","blockchain_226","blockchain_227","blockchain_228","blockchain_229","blockchain_230","blockchain_231","blockchain_232","blockchain_233","blockchain_234","blockchain_235","blockchain_236","blockchain_237","blockchain_238","blockchain_239","blockchain_240","blockchain_241","blockchain_242","blockchain_243","blockchain_244","blockchain_245","blockchain_246","blockchain_247","blockchain_248","blockchain_249","blockchain_250","blockchain_251","blockchain_252","blockchain_253","blockchain_254","blockchain_255","blockchain_256","blockchain_257","blockchain_258","blockchain_259","blockchain_260","blockchain_261","blockchain_262","blockchain_263","blockchain_264","blockchain_265","blockchain_266","blockchain_267","blockchain_268","blockchain_269","blockchain_270","blockchain_271","blockchain_272","blockchain_273","blockchain_274","blockchain_275","blockchain_276","blockchain_277","blockchain_278","blockchain_279","blockchain_280","blockchain_281","blockchain_282","blockchain_283","blockchain_284","blockchain_285","blockchain_286","blockchain_287","blockchain_288","blockchain_289","blockchain_290","blockchain_291","blockchain_292","blockchain_293","blockchain_294","blockchain_295","blockchain_296","blockchain_297","blockchain_298","blockchain_299","blockchain_300","blockchain_301","blockchain_302","blockchain_303","blockchain_304","blockchain_305","blockchain_306","blockchain_307","blockchain_308","blockchain_309","blockchain_310","blockchain_311","blockchain_312","blockchain_313","blockchain_314","blockchain_315","blockchain_316","blockchain_317","blockchain_318","blockchain_319","blockchain_320","blockchain_321","blockchain_322","blockchain_323","blockchain_324","blockchain_325","blockchain_326","blockchain_327","blockchain_328","blockchain_329","blockchain_330","blockchain_331","blockchain_332","blockchain_333","blockchain_334","blockchain_335","blockchain_336","blockchain_337","blockchain_338","blockchain_339","blockchain_340","blockchain_341","blockchain_342","blockchain_343","blockchain_344","blockchain_345","blockchain_346","blockchain_347","blockchain_348","blockchain_349","blockchain_350","blockchain_351","blockchain_352","blockchain_353","blockchain_354","blockchain_355","blockchain_356","blockchain_357","blockchain_358","blockchain_359","blockchain_360","blockchain_361","blockchain_362","blockchain_363","blockchain_364","blockchain_365","blockchain_366","blockchain_367","blockchain_368","blockchain_369","blockchain_370","blockchain_371","blockchain_372","blockchain_373","blockchain_374","blockchain_375","blockchain_376","blockchain_377","blockchain_378","blockchain_379","blockchain_380","blockchain_381","blockchain_382","blockchain_383","blockchain_384","blockchain_385","blockchain_386","blockchain_387","blockchain_388","blockchain_389","blockchain_390","blockchain_391","blockchain_392","blockchain_393","blockchain_394","blockchain_395","blockchain_396","blockchain_397","blockchain_398","blockchain_399","blockchain_400","blockchain_401","blockchain_402","blockchain_403","blockchain_404","blockchain_405","blockchain_406","blockchain_407","blockchain_408","blockchain_409","blockchain_410","blockchain_411","blockchain_412","blockchain_413","blockchain_414","blockchain_415","blockchain_416","blockchain_417","blockchain_418","blockchain_419","blockchain_420","blockchain_421","blockchain_422","blockchain_423","blockchain_424","blockchain_425","blockchain_426","blockchain_427","blockchain_428","blockchain_429","blockchain_430","blockchain_431","blockchain_432","blockchain_433","blockchain_434","blockchain_435","blockchain_436","blockchain_437","blockchain_438","blockchain_439","blockchain_440","blockchain_441","blockchain_442","blockchain_443","blockchain_444","blockchain_445","blockchain_446","blockchain_447","blockchain_448","blockchain_449","blockchain_450","blockchain_451","blockchain_452","blockchain_453","blockchain_454","blockchain_455","blockchain_456","blockchain_457","blockchain_458","blockchain_459","blockchain_460","blockchain_461","blockchain_462","blockchain_463","blockchain_464","blockchain_465","blockchain_466","blockchain_467","blockchain_468","blockchain_469","blockchain_470","blockchain_471","blockchain_472","blockchain_473","blockchain_474","blockchain_475","blockchain_476","blockchain_477","blockchain_478","blockchain_479","blockchain_480","blockchain_481","blockchain_482","blockchain_483","blockchain_484","blockchain_485","blockchain_486","blockchain_487","blockchain_488","blockchain_489","blockchain_490","blockchain_491","blockchain_492","blockchain_493","blockchain_494","blockchain_495","blockchain_496","blockchain_497","blockchain_498","blockchain_499"]
  
# # initialize the driver and load the page 
# for i in range(0,10000):
# 	blocklist = 'blockchain_' + str(i)
# 	webpagelist = 'https://www.etherchain.org/block/' + str(i)
# 	driver = webdriver.Chrome("C:\\Users\\raosa\\Desktop\\PROJECTS\\chromedriver.exe")
# 	driver.get(webpagelist)
# 	# save the page 
# 	time.sleep(10)
# 	pag.keyDown('ctrl')
# 	pag.press('s')
# 	pag.keyUp('ctrl')
# 	print('saving')
# 	time.sleep(3)
# 	pag.typewrite(blocklist)
# 	pag.press('enter')
# 	time.sleep(3)
# 	print(i)
# 	driver.close()