import numpy as np
import pandas as pd

usecols = ["stat_date", "vopenid", "uid", "onlinetime", "level", "upvote","onlinetime_list","corpid", "carteamid", "vclientip","country", "friend_num_plat", "friend_list_plat", "active_plat_num", "friend_num_game","friend_list_game", "active_game_num", "jiyounum", "lianrennum", "sidangnum", "guiminum", "jiyou", "lianren", "sidang", "guimi", "chg_intimacy", "avg_intimacy", "gender", "chat_num", "friend_chat_num", "send_gold_num", "draw_gold_num", "vote_num", "friend_vote_num", "give_gift_num", "get_gift_num", "recruit_num", "friend_recruit_num", "reply_recruit_num", "team_num", "leader_team_num", "invite_team_num", "apply_relation_num", "reply_relation_num", "del_friend_num", "invite_carteam_num", "friend_carteam_num", "carteam_leaader_num", "accept_carteam_num", "corps_active_change", "uid_corp_active_change", "corps_level_change", "suc_match_num", "create_corp_num", "signup_num", "achievement_num", "chat_uid_num", "friend_uid_num", "reject_friend_num", "del_friend_apply_num", "reject_relation_num", "del_intimacy_num", "exit_carteam_num", "kick_num", "bekick_num", "del_carteam_num", "transfer_carteam_num", "accept_invite_team_num", "reject_invite_team_num", "beaccept_invite_team_num", "bereject_invite_team_num", "beaccept_apply_team_num", "bereject_apply_team_num", "accept_apply_team_num", "reject_apply_team_num", "join_corp_num", "space_req_num", "space_bereq_num", "space_gift_num", "space_gift_unum", "login_num", "login_days", "is_comeback", "playerlang", "onlinetime_detail", "bevote_num", "corp_money_chg", "funny_mode_num", "funny_single_num", "funny_double_num", "funny_squad_num", "round_num", "avg_kill_count", "avg_hit_rate", "chicken_rate", "top10_rate", "avg_damage", "avg_box_num", "avg_survivaltime", "avg_moving", "head_shoot_rate", "avg_gun_kill", "avg_assist", "avg_healtimes", "avg_healamount", "avg_cure", "round_single_num", "round_double_num", "round_squad_num", "first_single_segment", "first_dou_segment", "first_squad_segment", "third_single_segment", "third_dou_segment", "third_squad_segment", "first_single_score", "first_dou_score", "first_squad_score",  "third_single_score", "third_dou_score", "third_squad_score", "systemhardware", "platid", "diamond_add_4week", "register_days", "content_sample", "ratingvalueafterchanged", "final_score", "diamond_add", "diamond_reduce_4week", "diamond_reduce", "classical_num", "funny_num", "tpp_num", "fpp_num", "solo_num",  "duo_num", "squad_num", "funny_quick_num","funny_sniper_num", "funny_war_num", "segment", "classical_team_rate",  "avg_collectindex", "avg_fps", "funny_team_rate", "is_register_month", "last_login_date", "avg_hit_distance", "reward_num", "share_num", "register_time", "is_etc", "is_vpn", "daily_buy_num", "game_start_num", "skin_num"] # 数据文件中相关的变量

# personal = ["onlinetime", "level", "upvote","signup_num", "achievement_num","login_num", "login_days", "round_num", "round_squad_num", "classical_num", "classical_team_rate", "funny_num", "funny_squad_num", "funny_quick_num", "funny_sniper_num", "funny_war_num", "funny_team_rate", "segment", "reward_num", "skin_num", "tpp_num", "fpp_num"] # 玩家基本信息
personal = ["onlinetime", "level", "upvote","signup_num", "achievement_num","login_num", "login_days", "round_num", "round_squad_num", "classical_num", "classical_team_rate", "funny_num", "funny_team_rate", "segment", "reward_num", "skin_num", "ip_counts"] # 玩家基本信息

relations = [ "friend_num_plat", "active_plat_num", "friend_num_game","active_game_num", "chat_num", "friend_chat_num", "send_gold_num", "draw_gold_num", "vote_num", "friend_vote_num","recruit_num", "friend_recruit_num", "reply_recruit_num","team_num", "leader_team_num", "invite_team_num", "apply_relation_num", "reply_relation_num", "del_friend_num","reject_friend_num", "del_friend_apply_num", "reject_relation_num", "del_intimacy_num", "accept_invite_team_num","reject_invite_team_num", "beaccept_invite_team_num", "bereject_invite_team_num", "beaccept_apply_team_num", "bereject_apply_team_num", "accept_apply_team_num", "reject_apply_team_num"] # 玩家互动信息

performance = ["avg_kill_count", "avg_hit_rate", "chicken_rate", "top10_rate", "avg_damage", "avg_box_num", "avg_survivaltime", "avg_moving", "head_shoot_rate", "avg_gun_kill", "avg_assist", "avg_healtimes", "avg_healamount", "avg_cure","avg_hit_distance", "final_score"] # 玩家战斗表现

diff_list = ["del_friend_num", "friend_num_game", "skin_num", "active_plat_num", "chat_num", "ip_counts", "round_squad_num", "level", "funny_num"] # 需要考虑纵向变化量的变量

def read_longitudinal_date(file_list, keep_shared = True, verbose = 0):
    '''
    read feature data of consecutive weeks
    keep_shared: only to keep users with uid appearing in every data file
    verbose: if verbose > 0, print data info 
    '''
    account_by_dates = []
    for file in file_list:
        df =  pd.read_csv(file, sep="\t",
                                    dtype={"stat_date":str, "vopenid":str, "uid":str,"corpid":str, "carteamid":str, 
                                        "friend_list_game":str, "friend_list_plat":str, 
                                        "jiyou":str,"lianren":str,"sidang":str,"guimi":str,
                                        "register_time":str, "last_login_date":str}, 
                                    usecols= usecols,
                                na_values=['-1', '\\N'])
        df.drop(np.where(df["uid"].apply(type) != str )[0], inplace=True)
        df = df[np.logical_not(df["uid"].duplicated())]
        if verbose:
            print(file, df.shape)
        account_by_dates.append(df)
        del(df)

    if keep_shared:
        shared_uid = account_by_dates[0]["uid"]
        for df in account_by_dates[1:]:
            shared_uid = np.intersect1d(shared_uid, df["uid"], assume_unique=True)
        if verbose:
            print("number of shared uid:", len(shared_uid))

        return [df.set_index("uid").loc[shared_uid] for df in account_by_dates]
    
    else:
        return [df.set_index("uid") for df in account_by_dates]