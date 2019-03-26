library(tidyselect)
library(tidyverse)

league <- read_csv(here::here("450_data/LeagueofLegends.csv"))




l2 <- league%>%
  mutate_at(function(x)map(x,codyR::np_erase),.vars = with_vars(colnames(league),contains("gold")))


gold <- read_csv(here::here("data/gold.csv"))%>%
  filter(!(Type %in% c("golddiff","goldblue","goldred")))

unique(gold$Type)

info <- read_csv(here::here("data/matchinfo.csv"))

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

