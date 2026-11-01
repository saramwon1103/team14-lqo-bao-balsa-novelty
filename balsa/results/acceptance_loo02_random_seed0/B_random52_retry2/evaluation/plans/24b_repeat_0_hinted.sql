/*+ HashJoin(chn rt k an mk cn mc mi it t ci n)
 MergeJoin(rt k an mk cn mc mi it t ci n)
 HashJoin(rt k an mk cn mc mi it t ci)
 MergeJoin(k an mk cn mc mi it t ci)
 HashJoin(an mk cn mc mi it t ci)
 NestLoop(mk cn mc mi it t ci)
 MergeJoin(mk cn mc mi it t)
 NestLoop(cn mc mi it t)
 HashJoin(cn mc mi it)
 HashJoin(mc mi it)
 NestLoop(mi it)
 SeqScan(chn)
 SeqScan(rt)
 IndexScan(k)
 SeqScan(an)
 SeqScan(mk)
 SeqScan(cn)
 IndexScan(mc)
 SeqScan(mi)
 IndexScan(it)
 IndexScan(t)
 IndexScan(ci)
 SeqScan(n)
 Leading((chn ((rt (k (an ((mk ((cn (mc (mi it))) t)) ci)))) n))) */
SELECT MIN(chn.name) AS voiced_char_name,
       MIN(n.name) AS voicing_actress_name,
       MIN(t.title) AS kung_fu_panda
FROM aka_name AS an,
     char_name AS chn,
     cast_info AS ci,
     company_name AS cn,
     info_type AS it,
     keyword AS k,
     movie_companies AS mc,
     movie_info AS mi,
     movie_keyword AS mk,
     name AS n,
     role_type AS rt,
     title AS t
WHERE ci.note IN ('(voice)',
                  '(voice: Japanese version)',
                  '(voice) (uncredited)',
                  '(voice: English version)')
  AND cn.country_code ='[us]'
  AND cn.name = 'DreamWorks Animation'
  AND it.info = 'release dates'
  AND k.keyword IN ('hero',
                    'martial-arts',
                    'hand-to-hand-combat',
                    'computer-animated-movie')
  AND mi.info IS NOT NULL
  AND (mi.info LIKE 'Japan:%201%'
       OR mi.info LIKE 'USA:%201%')
  AND n.gender ='f'
  AND n.name LIKE '%An%'
  AND rt.role ='actress'
  AND t.production_year > 2010
  AND t.title LIKE 'Kung Fu Panda%'
  AND t.id = mi.movie_id
  AND t.id = mc.movie_id
  AND t.id = ci.movie_id
  AND t.id = mk.movie_id
  AND mc.movie_id = ci.movie_id
  AND mc.movie_id = mi.movie_id
  AND mc.movie_id = mk.movie_id
  AND mi.movie_id = ci.movie_id
  AND mi.movie_id = mk.movie_id
  AND ci.movie_id = mk.movie_id
  AND cn.id = mc.company_id
  AND it.id = mi.info_type_id
  AND n.id = ci.person_id
  AND rt.id = ci.role_id
  AND n.id = an.person_id
  AND ci.person_id = an.person_id
  AND chn.id = ci.person_role_id
  AND k.id = mk.keyword_id;

