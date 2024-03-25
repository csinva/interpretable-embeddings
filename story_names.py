
def get_story_names(subject: str = "UTS01", train_or_test="train"):
    TRAIN_STORIES_01 = ['adollshouse', 'gpsformylostidentity', 'singlewomanseekingmanwich', 'adventuresinsayingyes', 'hangtime', 'sloth', 'afatherscover', 'haveyoumethimyet', 'souls', 'againstthewind', 'howtodraw', 'stagefright', 'alternateithicatom', 'ifthishaircouldtalk', 'stumblinginthedark', 'avatar', 'inamoment', 'superheroesjustforeachother', 'backsideofthestorm', 'itsabox', 'sweetaspie', 'becomingindian', 'jugglingandjesus', 'swimmingwithastronauts', 'beneaththemushroomcloud', 'kiksuya', 'thatthingonmyarm', 'birthofanation', 'leavingbaghdad', 'theadvancedbeginner', 'bluehope', 'legacy', 'theclosetthatateeverything', 'breakingupintheageofgoogle', 'lifeanddeathontheoregontrail', 'thecurse', 'buck', 'life', 'thefreedomridersandme', 'catfishingstrangerstofindmyself', 'lifereimagined', 'theinterview',
                        'cautioneating', 'listo', 'thepostmanalwayscalls', 'christmas1940', 'mayorofthefreaks', 'theshower', 'cocoonoflove', 'metsmagic', 'thetiniestbouquet', 'comingofageondeathrow', 'mybackseatviewofagreatromance', 'thetriangleshirtwaistconnection', 'exorcism', 'myfathershands', 'threemonths', 'eyespy', 'myfirstdaywiththeyankees', 'thumbsup', 'firetestforlove', 'naked', 'tildeath', 'food', 'notontheusualtour', 'treasureisland', 'forgettingfear', 'odetostepfather', 'undertheinfluence', 'onlyonewaytofindout', 'vixenandtheussr', 'gangstersandcookies', 'penpal', 'waitingtogo', 'goingthelibertyway', 'quietfire', 'whenmothersbullyback', 'goldiethegoldfish', 'reachingoutbetweenthebars', 'golfclubbing', 'shoppinginchina', 'wildwomenanddancingqueens']
    TRAIN_STORIES_02 = ['adollshouse', 'hangtime', 'sloth', 'adventuresinsayingyes', 'haveyoumethimyet', 'souls', 'afatherscover', 'howtodraw', 'stagefright', 'againstthewind', 'ifthishaircouldtalk', 'stumblinginthedark', 'alternateithicatom', 'inamoment', 'superheroesjustforeachother', 'avatar', 'itsabox', 'sweetaspie', 'backsideofthestorm', 'jugglingandjesus', 'swimmingwithastronauts', 'becomingindian', 'kiksuya', 'thatthingonmyarm', 'beneaththemushroomcloud', 'leavingbaghdad', 'theadvancedbeginner', 'birthofanation', 'legacy', 'theclosetthatateeverything', 'breakingupintheageofgoogle', 'lifeanddeathontheoregontrail', 'thecurse', 'buck', 'life', 'thefreedomridersandme', 'catfishingstrangerstofindmyself', 'lifereimagined', 'theinterview', 'cautioneating', 'listo',
                        'thepostmanalwayscalls', 'christmas1940', 'mayorofthefreaks', 'theshower', 'cocoonoflove', 'metsmagic', 'thetiniestbouquet', 'comingofageondeathrow', 'mybackseatviewofagreatromance', 'thetriangleshirtwaistconnection', 'exorcism', 'myfathershands', 'threemonths', 'eyespy', 'myfirstdaywiththeyankees', 'thumbsup', 'firetestforlove', 'naked', 'tildeath', 'food', 'notontheusualtour', 'treasureisland', 'forgettingfear', 'odetostepfather', 'undertheinfluence', 'onlyonewaytofindout', 'vixenandtheussr', 'gangstersandcookies', 'penpal', 'waitingtogo', 'goingthelibertyway', 'quietfire', 'whenmothersbullyback', 'goldiethegoldfish', 'reachingoutbetweenthebars', 'golfclubbing', 'shoppinginchina', 'wildwomenanddancingqueens']
    TRAIN_STORIES_03 = [
        'adollshouse', 'gpsformylostidentity', 'singlewomanseekingmanwich', 'adventuresinsayingyes', 'hangtime', 'sloth', 'afatherscover', 'haveyoumethimyet', 'souls', 'againstthewind', 'howtodraw', 'stagefright', 'alternateithicatom', 'ifthishaircouldtalk', 'stumblinginthedark', 'avatar', 'inamoment', 'superheroesjustforeachother', 'backsideofthestorm', 'itsabox', 'sweetaspie', 'becomingindian', 'jugglingandjesus', 'swimmingwithastronauts', 'beneaththemushroomcloud', 'kiksuya', 'thatthingonmyarm', 'birthofanation', 'leavingbaghdad', 'theadvancedbeginner', 'bluehope', 'legacy', 'theclosetthatateeverything', 'breakingupintheageofgoogle', 'lifeanddeathontheoregontrail', 'thecurse', 'buck', 'life', 'thefreedomridersandme', 'catfishingstrangerstofindmyself', 'lifereimagined', 'theinterview',
        'cautioneating', 'listo', 'thepostmanalwayscalls', 'christmas1940', 'mayorofthefreaks', 'theshower', 'cocoonoflove', 'metsmagic', 'thetiniestbouquet', 'comingofageondeathrow', 'mybackseatviewofagreatromance', 'thetriangleshirtwaistconnection', 'exorcism', 'myfathershands', 'threemonths', 'eyespy', 'myfirstdaywiththeyankees', 'thumbsup', 'firetestforlove', 'naked', 'tildeath', 'food', 'notontheusualtour', 'treasureisland', 'forgettingfear', 'odetostepfather', 'undertheinfluence', 'onlyonewaytofindout', 'vixenandtheussr', 'gangstersandcookies', 'penpal', 'waitingtogo', 'goingthelibertyway', 'quietfire', 'whenmothersbullyback', 'goldiethegoldfish', 'reachingoutbetweenthebars', 'golfclubbing', 'shoppinginchina', 'wildwomenanddancingqueens'
    ]
    TEST_STORIES = ["wheretheressmoke",
                    "fromboyhoodtofatherhood"]  # "onapproachtopluto"
    story_names_train = {
        "UTS01": TRAIN_STORIES_01,
        "UTS02": TRAIN_STORIES_02,
        "UTS03": TRAIN_STORIES_03,
    }
    if train_or_test == "train":
        return story_names_train[subject]
    elif train_or_test == "test":
        return TEST_STORIES


if __name__ == "__main__":
    for subject in ["UTS01", "UTS02", "UTS03"]:
        print(f"Subject: {subject}")
        print(
            f'len of train stories: {len(get_story_names(subject, "train"))}')
