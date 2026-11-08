/*+ HashJoin(mk rt ci cn mc t mi it n k chn an)
 HashJoin(mk rt ci cn mc t mi it n k chn)
 HashJoin(mk rt ci cn mc t mi it n k)
 HashJoin(mk rt ci cn mc t mi it n)
 HashJoin(rt ci cn mc t mi it n)
 HashJoin(ci cn mc t mi it n)
 HashJoin(ci cn mc t mi it)
 HashJoin(cn mc t mi it)
 HashJoin(mc t mi it)
 MergeJoin(t mi it)
 HashJoin(mi it)
 SeqScan(mk)
 SeqScan(rt)
 SeqScan(ci)
 SeqScan(cn)
 SeqScan(mc)
 IndexScan(t)
 IndexScan(mi)
 SeqScan(it)
 IndexScan(n)
 IndexScan(k)
 IndexScan(chn)
 IndexScan(an)
 Leading(((((mk (rt ((ci (cn (mc (t (mi it))))) n))) k) chn) an)) */
SELECT MIN(chn.name) AS voiced_char_name,
       MIN(n.name) AS voicing_actress_name,
       MIN(t.title) AS voiced_action_movie_jap_eng
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
  AND it.info = 'release dates'
  AND k.keyword IN ('hero',
                    'martial-arts',
                    'hand-to-hand-combat')
  AND mi.info IS NOT NULL
  AND (mi.info LIKE 'Japan:%201%'
       OR mi.info LIKE 'USA:%201%')
  AND n.gender ='f'
  AND n.name LIKE '%An%'
  AND rt.role ='actress'
  AND t.production_year > 2010
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

