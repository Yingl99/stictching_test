import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea

#prev 0.05/0.0173
def getImage(paths, zoom=0.0173):
    return OffsetImage(plt.imread(paths), zoom=zoom)


paths = ['F:\\img05\\JPG\\DJI_(1).JPG', 'F:\\img05\\JPG\\DJI_(10).JPG', 'F:\\img05\\JPG\\DJI_(100).JPG', 'F:\\img05\\JPG\\DJI_(101).JPG', 'F:\\img05\\JPG\\DJI_(102).JPG', 'F:\\img05\\JPG\\DJI_(103).JPG', 'F:\\img05\\JPG\\DJI_(104).JPG', 'F:\\img05\\JPG\\DJI_(105).JPG', 'F:\\img05\\JPG\\DJI_(106).JPG', 'F:\\img05\\JPG\\DJI_(107).JPG', 'F:\\img05\\JPG\\DJI_(108).JPG', 'F:\\img05\\JPG\\DJI_(109).JPG', 'F:\\img05\\JPG\\DJI_(11).JPG', 'F:\\img05\\JPG\\DJI_(110).JPG', 'F:\\img05\\JPG\\DJI_(111).JPG', 'F:\\img05\\JPG\\DJI_(112).JPG', 'F:\\img05\\JPG\\DJI_(113).JPG', 'F:\\img05\\JPG\\DJI_(114).JPG', 'F:\\img05\\JPG\\DJI_(115).JPG', 'F:\\img05\\JPG\\DJI_(116).JPG', 'F:\\img05\\JPG\\DJI_(117).JPG', 'F:\\img05\\JPG\\DJI_(118).JPG', 'F:\\img05\\JPG\\DJI_(119).JPG', 'F:\\img05\\JPG\\DJI_(12).JPG', 'F:\\img05\\JPG\\DJI_(120).JPG', 'F:\\img05\\JPG\\DJI_(121).JPG', 'F:\\img05\\JPG\\DJI_(122).JPG', 'F:\\img05\\JPG\\DJI_(123).JPG', 'F:\\img05\\JPG\\DJI_(124).JPG', 'F:\\img05\\JPG\\DJI_(125).JPG', 'F:\\img05\\JPG\\DJI_(126).JPG', 'F:\\img05\\JPG\\DJI_(127).JPG', 'F:\\img05\\JPG\\DJI_(128).JPG', 'F:\\img05\\JPG\\DJI_(129).JPG', 'F:\\img05\\JPG\\DJI_(13).JPG', 'F:\\img05\\JPG\\DJI_(130).JPG', 'F:\\img05\\JPG\\DJI_(131).JPG', 'F:\\img05\\JPG\\DJI_(132).JPG', 'F:\\img05\\JPG\\DJI_(133).JPG', 'F:\\img05\\JPG\\DJI_(134).JPG', 'F:\\img05\\JPG\\DJI_(135).JPG', 'F:\\img05\\JPG\\DJI_(136).JPG', 'F:\\img05\\JPG\\DJI_(137).JPG', 'F:\\img05\\JPG\\DJI_(138).JPG', 'F:\\img05\\JPG\\DJI_(139).JPG', 'F:\\img05\\JPG\\DJI_(14).JPG', 'F:\\img05\\JPG\\DJI_(140).JPG', 'F:\\img05\\JPG\\DJI_(141).JPG', 'F:\\img05\\JPG\\DJI_(142).JPG', 'F:\\img05\\JPG\\DJI_(143).JPG', 'F:\\img05\\JPG\\DJI_(144).JPG', 'F:\\img05\\JPG\\DJI_(145).JPG', 'F:\\img05\\JPG\\DJI_(146).JPG', 'F:\\img05\\JPG\\DJI_(147).JPG', 'F:\\img05\\JPG\\DJI_(148).JPG', 'F:\\img05\\JPG\\DJI_(149).JPG', 'F:\\img05\\JPG\\DJI_(15).JPG', 'F:\\img05\\JPG\\DJI_(150).JPG', 'F:\\img05\\JPG\\DJI_(151).JPG', 'F:\\img05\\JPG\\DJI_(152).JPG', 'F:\\img05\\JPG\\DJI_(153).JPG', 'F:\\img05\\JPG\\DJI_(154).JPG', 'F:\\img05\\JPG\\DJI_(155).JPG', 'F:\\img05\\JPG\\DJI_(156).JPG', 'F:\\img05\\JPG\\DJI_(157).JPG', 'F:\\img05\\JPG\\DJI_(158).JPG', 'F:\\img05\\JPG\\DJI_(159).JPG', 'F:\\img05\\JPG\\DJI_(16).JPG', 'F:\\img05\\JPG\\DJI_(160).JPG', 'F:\\img05\\JPG\\DJI_(161).JPG', 'F:\\img05\\JPG\\DJI_(162).JPG', 'F:\\img05\\JPG\\DJI_(163).JPG', 'F:\\img05\\JPG\\DJI_(164).JPG', 'F:\\img05\\JPG\\DJI_(17).JPG', 'F:\\img05\\JPG\\DJI_(18).JPG', 'F:\\img05\\JPG\\DJI_(19).JPG', 'F:\\img05\\JPG\\DJI_(2).JPG', 'F:\\img05\\JPG\\DJI_(20).JPG', 'F:\\img05\\JPG\\DJI_(21).JPG', 'F:\\img05\\JPG\\DJI_(22).JPG', 'F:\\img05\\JPG\\DJI_(23).JPG', 'F:\\img05\\JPG\\DJI_(24).JPG', 'F:\\img05\\JPG\\DJI_(25).JPG', 'F:\\img05\\JPG\\DJI_(26).JPG', 'F:\\img05\\JPG\\DJI_(27).JPG', 'F:\\img05\\JPG\\DJI_(28).JPG', 'F:\\img05\\JPG\\DJI_(29).JPG', 'F:\\img05\\JPG\\DJI_(3).JPG', 'F:\\img05\\JPG\\DJI_(30).JPG', 'F:\\img05\\JPG\\DJI_(31).JPG', 'F:\\img05\\JPG\\DJI_(32).JPG', 'F:\\img05\\JPG\\DJI_(33).JPG', 'F:\\img05\\JPG\\DJI_(34).JPG', 'F:\\img05\\JPG\\DJI_(35).JPG', 'F:\\img05\\JPG\\DJI_(36).JPG', 'F:\\img05\\JPG\\DJI_(37).JPG', 'F:\\img05\\JPG\\DJI_(38).JPG', 'F:\\img05\\JPG\\DJI_(39).JPG', 'F:\\img05\\JPG\\DJI_(4).JPG', 'F:\\img05\\JPG\\DJI_(40).JPG', 'F:\\img05\\JPG\\DJI_(41).JPG', 'F:\\img05\\JPG\\DJI_(42).JPG', 'F:\\img05\\JPG\\DJI_(43).JPG', 'F:\\img05\\JPG\\DJI_(44).JPG', 'F:\\img05\\JPG\\DJI_(45).JPG', 'F:\\img05\\JPG\\DJI_(46).JPG', 'F:\\img05\\JPG\\DJI_(47).JPG', 'F:\\img05\\JPG\\DJI_(48).JPG', 'F:\\img05\\JPG\\DJI_(49).JPG', 'F:\\img05\\JPG\\DJI_(5).JPG', 'F:\\img05\\JPG\\DJI_(50).JPG', 'F:\\img05\\JPG\\DJI_(51).JPG', 'F:\\img05\\JPG\\DJI_(52).JPG', 'F:\\img05\\JPG\\DJI_(53).JPG', 'F:\\img05\\JPG\\DJI_(54).JPG', 'F:\\img05\\JPG\\DJI_(55).JPG', 'F:\\img05\\JPG\\DJI_(56).JPG', 'F:\\img05\\JPG\\DJI_(57).JPG', 'F:\\img05\\JPG\\DJI_(58).JPG', 'F:\\img05\\JPG\\DJI_(59).JPG', 'F:\\img05\\JPG\\DJI_(6).JPG', 'F:\\img05\\JPG\\DJI_(60).JPG', 'F:\\img05\\JPG\\DJI_(61).JPG', 'F:\\img05\\JPG\\DJI_(62).JPG', 'F:\\img05\\JPG\\DJI_(63).JPG', 'F:\\img05\\JPG\\DJI_(64).JPG', 'F:\\img05\\JPG\\DJI_(65).JPG', 'F:\\img05\\JPG\\DJI_(66).JPG', 'F:\\img05\\JPG\\DJI_(67).JPG', 'F:\\img05\\JPG\\DJI_(68).JPG', 'F:\\img05\\JPG\\DJI_(69).JPG', 'F:\\img05\\JPG\\DJI_(7).JPG', 'F:\\img05\\JPG\\DJI_(70).JPG', 'F:\\img05\\JPG\\DJI_(71).JPG', 'F:\\img05\\JPG\\DJI_(72).JPG', 'F:\\img05\\JPG\\DJI_(73).JPG', 'F:\\img05\\JPG\\DJI_(74).JPG', 'F:\\img05\\JPG\\DJI_(75).JPG', 'F:\\img05\\JPG\\DJI_(76).JPG', 'F:\\img05\\JPG\\DJI_(77).JPG', 'F:\\img05\\JPG\\DJI_(78).JPG', 'F:\\img05\\JPG\\DJI_(79).JPG', 'F:\\img05\\JPG\\DJI_(8).JPG', 'F:\\img05\\JPG\\DJI_(80).JPG', 'F:\\img05\\JPG\\DJI_(81).JPG', 'F:\\img05\\JPG\\DJI_(82).JPG', 'F:\\img05\\JPG\\DJI_(83).JPG', 'F:\\img05\\JPG\\DJI_(84).JPG', 'F:\\img05\\JPG\\DJI_(85).JPG', 'F:\\img05\\JPG\\DJI_(86).JPG', 'F:\\img05\\JPG\\DJI_(87).JPG', 'F:\\img05\\JPG\\DJI_(88).JPG', 'F:\\img05\\JPG\\DJI_(89).JPG', 'F:\\img05\\JPG\\DJI_(9).JPG', 'F:\\img05\\JPG\\DJI_(90).JPG', 'F:\\img05\\JPG\\DJI_(91).JPG', 'F:\\img05\\JPG\\DJI_(92).JPG', 'F:\\img05\\JPG\\DJI_(93).JPG', 'F:\\img05\\JPG\\DJI_(94).JPG', 'F:\\img05\\JPG\\DJI_(95).JPG', 'F:\\img05\\JPG\\DJI_(96).JPG', 'F:\\img05\\JPG\\DJI_(97).JPG', 'F:\\img05\\JPG\\DJI_(98).JPG', 'F:\\img05\\JPG\\DJI_(99).JPG']

x = [120.9067704, 120.9046745, 120.8724655, 120.871616, 120.8707654, 120.8699214, 120.8699192, 120.8690743, 120.8687086, 120.8634581, 120.8642995, 120.8651477, 120.9038271, 120.8659949, 120.8668425, 120.8676905, 120.8685373, 120.8693862, 120.8702335, 120.8710818, 120.871929, 120.8727777, 120.8732739, 120.9029807, 120.869005, 120.8681476, 120.867302, 120.8664547, 120.8656062, 120.8655994, 120.8647592, 120.8639098, 120.8630623, 120.8622136, 120.9024484, 120.8613652, 120.8605171, 120.8596748, 120.8566445, 120.8574863, 120.8583341, 120.8591814, 120.8600284, 120.8608767, 120.8608702, 120.8981215, 120.8617238, 120.8625714, 120.863419, 120.8642671, 120.8647321, 120.8597024, 120.8588479, 120.858001, 120.8571528, 120.8563048, 120.8989608, 120.8554551, 120.8546096, 120.8537583, 120.8537076, 120.8507846, 120.8516259, 120.8524732, 120.8533216, 120.853323, 120.8541687, 120.8989594, 120.855017, 120.8554915, 120.8502953, 120.8494419, 120.8486628, 120.8998079, 120.9006546, 120.9015032, 120.9076259, 120.9023506, 120.9031976, 120.9040067, 120.9005178, 120.8996664, 120.8988194, 120.897971, 120.8971248, 120.8962765, 120.8954292, 120.9084734, 120.8945805, 120.8937326, 120.892885, 120.8920379, 120.8919079, 120.8883459, 120.8891868, 120.8900339, 120.890882, 120.8908832, 120.9093203, 120.8917288, 120.8925754, 120.8934249, 120.8942715, 120.8951191, 120.8959678, 120.8967818, 120.8930601, 120.8922084, 120.8913611, 120.9101676, 120.8905141, 120.8896664, 120.8888197, 120.8879724, 120.8871216, 120.8862764, 120.8854283, 120.8847733, 120.8847717, 120.8812088, 120.9103341, 120.88205, 120.8828969, 120.8837459, 120.8845929, 120.8854403, 120.8862882, 120.8871363, 120.8879835, 120.8888302, 120.8893258, 120.9072211, 120.8854706, 120.8846193, 120.8837711, 120.8829238, 120.8820771, 120.8812263, 120.8803786, 120.8803693, 120.8795264, 120.8786787, 120.9063699, 120.8778327, 120.8776308, 120.8735794, 120.874422, 120.8752693, 120.8761179, 120.8769651, 120.8778132, 120.8786607, 120.8795078, 120.9055225, 120.8803562, 120.8812036, 120.8812039, 120.8815357, 120.8775563, 120.8767048, 120.8758555, 120.8750085, 120.8741625, 120.8733145]

y = [31.05865236, 31.06506589, 31.06767372, 31.06498633, 31.06229839, 31.05961064, 31.05961286, 31.0569245, 31.05577617, 31.05220136, 31.05488092, 31.05756803, 31.06237931, 31.06025594, 31.06294403, 31.06563075, 31.06832022, 31.07100739, 31.07369528, 31.07638483, 31.07907311, 31.08176003, 31.08333578, 31.05969133, 31.08287781, 31.08019206, 31.07750208, 31.07481336, 31.07212567, 31.07213625, 31.06943842, 31.0667495, 31.06406147, 31.06137289, 31.05800572, 31.05868483, 31.0559965, 31.053325, 31.05678839, 31.05947603, 31.062165, 31.06485247, 31.06754083, 31.07022925, 31.07022803, 31.05735319, 31.07291889, 31.07560653, 31.07829375, 31.08098233, 31.08245522, 31.07959806, 31.07690842, 31.07421872, 31.07153017, 31.06884161, 31.06003511, 31.06615219, 31.06346322, 31.06077678, 31.06060689, 31.06441603, 31.06710428, 31.06979311, 31.07248189, 31.07248489, 31.07516953, 31.06003144, 31.07785819, 31.07936236, 31.07598081, 31.07329106, 31.07081406, 31.06272044, 31.06540644, 31.06809431, 31.06134067, 31.07078014, 31.07346711, 31.07603103, 31.07803478, 31.07535125, 31.07266483, 31.06997844, 31.067286, 31.06459808, 31.06191433, 31.06402661, 31.05922825, 31.05654131, 31.05385486, 31.05116758, 31.05075883, 31.05253411, 31.05521897, 31.05790592, 31.06059319, 31.0605945, 31.06671308, 31.06327917, 31.06596628, 31.0686555, 31.07134142, 31.07402886, 31.07671494, 31.07930011, 31.08056961, 31.07788697, 31.07519872, 31.06939992, 31.07250533, 31.06982542, 31.06712875, 31.06445067, 31.06176294, 31.05907581, 31.05638706, 31.05431186, 31.05431356, 31.05608603, 31.06992742, 31.0587705, 31.06145858, 31.06414647, 31.06683447, 31.06952167, 31.07220925, 31.07489561, 31.07758319, 31.08027122, 31.08183589, 31.07311986, 31.08268767, 31.08000456, 31.07731056, 31.07462072, 31.07193411, 31.0692475, 31.06656192, 31.06655022, 31.06387894, 31.06118983, 31.07043878, 31.05850206, 31.0578635, 31.05810061, 31.06078206, 31.06346872, 31.06615714, 31.06884422, 31.07153208, 31.07421983, 31.07690736, 31.06775281, 31.07959544, 31.08228258, 31.08228189, 31.0833325, 31.0838005, 31.08111461, 31.078427, 31.07573733, 31.07305081, 31.07036103]


fig = plt.figure(figsize=(20, 14), dpi=72)
ax = fig.add_subplot(111)
ax.scatter(x, y)

for x0, y0, path in zip(x, y, paths):
    ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
    ax.add_artist(ab)

# for x, y, z in zip(x, y, paths):
#     ax.annotate('({})'.format(z), xy=(x, y), fontsize=15)
# for x, y in zip(x, y):
#     ax.annotate('({}, {})'.format(x, y), xy=(x, y), fontsize=22)
# label = "{:.5f}".format(x, y)
# ac = AnnotationBbox(label, xy)
# plt.annotate(label, (x,y), textcoord="offset points", xytext=(20,20), ha="center")
# plt.imshow(ax, origin='lower', extent=[-56.444444444, 56.444444444, -45.861111111, 45.861111111], aspect=1)

plt.axis("equal")
plt.savefig('F:\\test\\figure16.jpg', dpi=800, format='jpg')
plt.show()

