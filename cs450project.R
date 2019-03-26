library(tidyselect)
library(tidyverse)

league <- read_csv(here::here("450_data/LeagueofLegends.csv"))




l2 <- league%>%
  mutate_at(function(x)map(x,codyR::np_erase),.vars = with_vars(colnames(league),contains("gold")))


gold <- read_csv(here::here("data/gold.csv"))%>%
  filter(!(Type %in% c("golddiff","goldblue","goldred")))

unique(gold$Type)

info <- read_csv(here::here("data/matchinfo.csv"))

positions <- info%>%
  gather(`blueTop`,`blueJungle`,`blueMiddle`,`blueADC`,`blueSupport`,`redTop`,`redJungle`,`redMiddle`,`redADC`,`redSupport`,key = "position",value = "name")%>%
  select(Address,position,name)

gold_dif <- gold%>%
  gather(time,contains('min'),value = gold)%>%
  mutate(time = as.numeric(str_remove_all(time,"\\D")))%>%
  group_by(Address,time)%>%
  spread(Type,value = gold)%>%
  transmute(blue_adc_dif = goldblueADC-goldredADC,
            blue_jung_dif = goldblueJungle-goldredJungle,
            blue_sup_dif = goldblueSupport-goldredSupport,
            blue_mid_dif = goldblueMiddle - goldredMiddle,
            blue_top_dif = goldblueTop - goldredTop,
            red_adc_dif = blue_adc_dif*-1,
            red_jung_dif = blue_jung_dif*-1,
            red_sup_dif = blue_sup_dif*-1,
            red_mid_dif = blue_mid_dif*-1,
            red_top_dif = blue_top_dif*-1,
            )

kills <- read_csv(here::here("data/kills.csv"))%>%
  filter(!(Victim %in% c("TooEarly","None")))%>%
  separate(Victim, sep= " ", extra = "merge", into = c("Team","Player"))%>%
  mutate(dead_player = case_when(
    is.na(Player) ~ Team,
  T ~ Player
  ),
    dead_team = case_when(
    is.na(Player) ~ "none",
    T ~ Team
  ))%>%
  select(-Player,-Team)%>%
  separate(Killer, sep= " ", extra = "merge", into = c("Team","Player"))%>%
  mutate(kill_player = case_when(
    is.na(Player) ~ Team,
    T ~ Player
  ),
  kill_team = case_when(
    is.na(Player) ~ "none",
    T ~ Team
  ))%>%
  select(-Player)%>%
  {mutate_at(.,function(x)map(x,function(y)
     ifelse(is.na(unlist(str_split(y," "))[2]),
            unlist(str_split(y," "))[1],
            unlist(str_split(y," "))[2])),
     .vars = with_vars(colnames(.),contains("Assist")))}


kill <- left_join(kills,positions,by = c("kill_player"="name","Address"))%>%
  group_by(kill_player,Address)%>%
  summarize(kills = n())

death <- left_join(kills,positions,by = c("dead_player"="name","Address"))%>%
  group_by(dead_player,Address)%>%
  summarize(deaths = n())

assists <- kills%>%
  gather(contains("Assist"),key = "useless",value = "assist")%>%
  mutate(assist = unlist(assist))%>%
  filter(!is.na(assist))%>%
  left_join(positions, by = c("assist" ="name","Address"))%>%
  group_by(assist,Address)%>%
  summarize(assists = n())

kda <- full_join(kill,death, by = c("kill_player" = "dead_player", "Address"))%>%
  full_join(assists, by = c("kill_player" = "assist", "Address"))%>%
  mutate_at(.vars =c(3,4,5),function(x)map(x,function(y) ifelse(is.na(y),0,y)))
    
